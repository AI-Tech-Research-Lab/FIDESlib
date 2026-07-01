//
// Ciphertext offload/reload (device <-> host) round-trip tests.
//

#include <openfhe.h>
#undef duration
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace FIDESlib::Testing {
class CiphertextOffloadTest : public GeneralParametrizedTest {
  protected:
	FIDESlib::CKKS::Context& SetupGPUContext() {
		cc->Enable(lbcrypto::PKE);
		cc->Enable(lbcrypto::KEYSWITCH);
		cc->Enable(lbcrypto::LEVELEDSHE);

		FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
		FIDESlib::CKKS::Context& cc_		= GPUcc;
		cc_									= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
		return cc_;
	}
};

// 1. Bit-exact identity round trip: offload()+reload() must reproduce the exact same
// RNS limbs and metadata as a plain store(), since only raw device<->host copies and
// dealloc/realloc are involved (no decrypt/rescale/level-switch/NTT).
TEST_P(CiphertextOffloadTest, BitExactRoundTrip) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1									= cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw1		= FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);

	FIDESlib::CKKS::RawCipherText before;
	GPUct.store(before);

	ASSERT_FALSE(GPUct.isOffloaded());
	GPUct.offload();
	ASSERT_TRUE(GPUct.isOffloaded());
	GPUct.reload();
	ASSERT_FALSE(GPUct.isOffloaded());

	FIDESlib::CKKS::RawCipherText after;
	GPUct.store(after);

	ASSERT_EQ(before.sub_0, after.sub_0);
	ASSERT_EQ(before.sub_1, after.sub_1);
	ASSERT_EQ(before.NoiseLevel, after.NoiseLevel);
	ASSERT_EQ(before.Noise, after.Noise);
	ASSERT_EQ(before.slots, after.slots);
	ASSERT_EQ(before.keyid, after.keyid);
	ASSERT_EQ(before.numRes, after.numRes);
}

// 2. Correctness after reload: decrypting a reloaded ciphertext must match the original
// plaintext, using the same tolerance-based comparison the rest of the suite uses.
TEST_P(CiphertextOffloadTest, DecryptAfterReload) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1									= cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw1		= FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);

	GPUct.offload();
	GPUct.reload();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct.store(raw_res);
	auto cResGPU = c1->Clone();
	GetOpenFHECipherText(cResGPU, raw_res);

	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	resultGPU->SetLength(generalTestParams.batchSize);

	lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, c1, &result);
	result->SetLength(generalTestParams.batchSize);

	ASSERT_ERROR_OK(result, resultGPU);
}

// 3. Compute-after-reload: prove the reloaded ciphertext is fully functional by running a
// GPU add and comparing against the CPU (OpenFHE) result.
TEST_P(CiphertextOffloadTest, ComputeAfterReload) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	auto c1									= cc->Encrypt(keys.publicKey, ptxt1);
	auto c2									= cc->Encrypt(keys.publicKey, ptxt2);
	FIDESlib::CKKS::RawCipherText raw1		= FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2		= FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw2);

	GPUct1.offload();
	GPUct2.offload();
	GPUct2.reload(); // an offloaded ciphertext used as a read-only operand is the caller's
					  // responsibility to reload; only the receiver auto-reloads (see
					  // ensureResident()'s doc comment).

	// add() must transparently reload `this` (GPUct1) rather than crash on freed pointers.
	GPUct1.add(GPUct2);
	ASSERT_FALSE(GPUct1.isOffloaded());

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto cResGPU = c1->Clone();
	GetOpenFHECipherText(cResGPU, raw_res);

	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	resultGPU->SetLength(generalTestParams.batchSize);

	auto cAdd = cc->EvalAdd(c1, c2);
	lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, cAdd, &result);
	result->SetLength(generalTestParams.batchSize);

	ASSERT_ERROR_OK(result, resultGPU);
}

// 4. VRAM is actually released. FIDESlib's GPU allocator (GPUmalloc/GPUfree in
// CudaUtils.cu) recycles same-size buffers through a process-local free-list pool
// instead of returning memory to the CUDA driver, so a single offload() will not show
// up as an *increase* in cudaMemGetInfo's free byte count (the freed limb goes back to
// the pool, not to the driver). What we *can* observe: if freeGPU()/dropLimb() failed
// to release the old limbs (a leak), repeatedly offloading+reloading same-shaped
// ciphertexts would keep requesting brand new pool chunks and free VRAM would keep
// shrinking. Absence of that continued shrinkage after a brief warm-up is the
// equivalent, allocator-aware way to prove the memory is actually being released.
TEST_P(CiphertextOffloadTest, VRAMReleasedNoLeakAcrossCycles) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1									= cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw1		= FIDESlib::CKKS::GetRawCipherText(cc, c1);

	constexpr int kWarmupCycles = 4;
	constexpr int kMeasuredCycles = 40;

	auto cycle = [&]() {
		FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
		GPUct.offload();
		cudaDeviceSynchronize();
		GPUct.reload();
		cudaDeviceSynchronize();
	};

	// Warm up: first-touch of each distinct limb byte-size pulls a fresh slab from the
	// driver; subsequent cycles of the same size should just recycle pool entries.
	for (int i = 0; i < kWarmupCycles; ++i) {
		cycle();
	}

	size_t free_before = 0, total = 0;
	cudaMemGetInfo(&free_before, &total);

	for (int i = 0; i < kMeasuredCycles; ++i) {
		cycle();
	}

	size_t free_after = 0;
	cudaMemGetInfo(&free_after, &total);

	// Allow a small amount of slack for unrelated allocator bookkeeping, but a real leak
	// across dozens of same-shaped alloc/free cycles would dwarf this tolerance.
	constexpr long long kToleranceBytes = 16ll * 1024 * 1024;
	ASSERT_GE(static_cast<long long>(free_after) + kToleranceBytes, static_cast<long long>(free_before));
}

// 5. Idempotence / boundaries: double offload, double reload, offloading a level-0
// ciphertext, and offloading one that has already been dropped to a low level.
TEST_P(CiphertextOffloadTest, IdempotentAndBoundaryLevels) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	auto c1									= cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw1		= FIDESlib::CKKS::GetRawCipherText(cc, c1);

	// Double offload / double reload are no-ops.
	{
		FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
		FIDESlib::CKKS::RawCipherText before;
		GPUct.store(before);

		GPUct.offload();
		GPUct.offload();
		ASSERT_TRUE(GPUct.isOffloaded());
		GPUct.reload();
		GPUct.reload();
		ASSERT_FALSE(GPUct.isOffloaded());

		FIDESlib::CKKS::RawCipherText after;
		GPUct.store(after);
		ASSERT_EQ(before.sub_0, after.sub_0);
		ASSERT_EQ(before.sub_1, after.sub_1);
	}

	// Level-0 (single-limb) ciphertext round trip.
	{
		FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
		GPUct.dropToLevel(0, true);
		ASSERT_EQ(GPUct.getLevel(), 0);

		FIDESlib::CKKS::RawCipherText before;
		GPUct.store(before);

		GPUct.offload();
		GPUct.reload();
		ASSERT_EQ(GPUct.getLevel(), 0);

		FIDESlib::CKKS::RawCipherText after;
		GPUct.store(after);
		ASSERT_EQ(before.sub_0, after.sub_0);
		ASSERT_EQ(before.sub_1, after.sub_1);
	}

	// A ciphertext already dropped to a low (but not 0) level.
	if (GPUcc->L >= 2) {
		FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
		GPUct.dropToLevel(1, true);
		ASSERT_EQ(GPUct.getLevel(), 1);

		FIDESlib::CKKS::RawCipherText before;
		GPUct.store(before);

		GPUct.offload();
		GPUct.reload();
		ASSERT_EQ(GPUct.getLevel(), 1);

		FIDESlib::CKKS::RawCipherText after;
		GPUct.store(after);
		ASSERT_EQ(before.sub_0, after.sub_0);
		ASSERT_EQ(before.sub_1, after.sub_1);
	}
}

INSTANTIATE_TEST_SUITE_P(CiphertextOffloadTests, CiphertextOffloadTest, testing::Values(TTALL64));

} // namespace FIDESlib::Testing
