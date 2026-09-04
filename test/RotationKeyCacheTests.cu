//
// Rotation-key VRAM cache tests: lazy key creation, on-demand reload, LRU eviction
// under a byte budget, manual offload, pinning, and bit-exactness of rotations across
// offload/reload round trips.
//

#include <openfhe.h>
#undef duration
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace FIDESlib::Testing {

class RotationKeyCacheTest : public GeneralParametrizedTest {
  protected:
	FIDESlib::CKKS::Context& SetupGPUContext() {
		cc->Enable(lbcrypto::PKE);
		cc->Enable(lbcrypto::KEYSWITCH);
		cc->Enable(lbcrypto::LEVELEDSHE);

		// The GPU context is cached & shared across tests with equal Parameters, so start
		// every test from a clean, order-independent cache configuration. Also (re)generate
		// the CPU-side automorphism keys: TearDown clears them between tests.
		cc->EvalRotateKeyGen(keys.secretKey, {1, 2, 3, 4, 5, 6, 7, 8});

		FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
		FIDESlib::CKKS::Context& cc_		= GPUcc;
		cc_									= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
		cc_->SetRotationKeyCache(SIZE_MAX);
		return cc_;
	}

	void TearDown() override {
		// Don't leak the finite budget into other test suites sharing this context.
		if (GPUcc) {
			GPUcc->SetRotationKeyCache(SIZE_MAX);
		}
		GeneralParametrizedTest::TearDown();
	}

	/** Registers rotation keys for indexes 1..4. If `lazy`, the context must already have
	 * a finite cache budget (keys are then created host-only). Returns one key's VRAM size
	 * (all keys share the same shape, hence the same size). */
	size_t AddRotationKeys(FIDESlib::CKKS::Context& cc_, const bool lazy) {
		size_t one_key_bytes = 0;
		for (const int i : {1, 2, 3, 4}) {
			FIDESlib::CKKS::RawKeySwitchKey raw = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, i, cc);
			FIDESlib::CKKS::KeySwitchingKey ksk(cc_);
			ksk.Initialize(raw, lazy);
			one_key_bytes = ksk.limbBytes();
			EXPECT_GT(one_key_bytes, (size_t)0);
			cc_->AddRotationKey(i, std::move(ksk));
		}
		return one_key_bytes;
	}

	FIDESlib::CKKS::Ciphertext MakeGpuCiphertext(FIDESlib::CKKS::Context& cc_, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cpu_ct) {
		FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, cpu_ct);
		return FIDESlib::CKKS::Ciphertext(cc_, raw);
	}

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> MakePlaintextCiphertext() {
		std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
		return cc->Encrypt(keys.publicKey, ptxt1);
	}

	/** Decrypts `gpu_res` (a rotation of `cpu_ct` by `index`) and checks it against the
	 * CPU (OpenFHE) EvalRotate result. */
	void ExpectRotationMatchesOpenFHE(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cpu_ct,
		const FIDESlib::CKKS::Ciphertext& gpu_res, const int index) {
		FIDESlib::CKKS::RawCipherText raw_res;
		const_cast<FIDESlib::CKKS::Ciphertext&>(gpu_res).store(raw_res);

		auto cResGPU = cpu_ct->Clone();
		GetOpenFHECipherText(cResGPU, raw_res);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		resultGPU->SetLength(generalTestParams.batchSize);

		auto expected_cpu = cc->EvalRotate(cpu_ct, index);
		lbcrypto::Plaintext expected;
		cc->Decrypt(keys.secretKey, expected_cpu, &expected);
		expected->SetLength(generalTestParams.batchSize);

		ASSERT_ERROR_OK(expected, resultGPU);
	}
};

// 1. With a finite budget set BEFORE key registration, creating a full set of rotation
// keys must not allocate any VRAM for them: keys exist only as host RAM snapshots and
// report as not resident.
TEST_P(RotationKeyCacheTest, LazyCreationDoesNotTouchVRAM) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40); // finite => lazy creation mode
	size_t one_key = AddRotationKeys(cc_, true);

	for (const int i : {1, 2, 3, 4}) {
		ASSERT_FALSE(cc_->IsRotationKeyResident(i));
	}
	ASSERT_EQ(cc_->RotationKeyCacheResidentBytes(), (size_t)0);
	ASSERT_GT(one_key, (size_t)0);

	cudaDeviceSynchronize();
	size_t free_before = 0, total = 0;
	cudaMemGetInfo(&free_before, &total);

	// Creating more lazy keys must still not allocate VRAM.
	for (const int i : {5, 6, 7, 8}) {
		FIDESlib::CKKS::RawKeySwitchKey raw = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, i, cc);
		FIDESlib::CKKS::KeySwitchingKey ksk(cc_);
		ksk.Initialize(raw, true);
		cc_->AddRotationKey(i, std::move(ksk));
		ASSERT_FALSE(cc_->IsRotationKeyResident(i));
	}
	cudaDeviceSynchronize();

	size_t free_after = 0;
	cudaMemGetInfo(&free_after, &total);
	// Small slack for allocator bookkeeping; one key would be ~2 * (L+1+dnum) * N * 8.
	constexpr long long kSlackBytes = 64ll * 1024 * 1024;
	ASSERT_GE(static_cast<long long>(free_after) + kSlackBytes, static_cast<long long>(free_before));
	ASSERT_EQ(cc_->RotationKeyCacheResidentBytes(), (size_t)0);
}

// 2. With a budget of exactly one key, rotating with each of the four keys must
// transparently load it, produce correct results, and keep the cache at <= one key.
TEST_P(RotationKeyCacheTest, BudgetOfOneKeyRotationsAreCorrect) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40);
	size_t one_key = AddRotationKeys(cc_, true);
	cc_->SetRotationKeyCache(one_key); // shrink to exactly one key

	auto c1 = MakePlaintextCiphertext();
	for (const int i : {1, 2, 3, 4}) {
		FIDESlib::CKKS::Ciphertext GPUct = MakeGpuCiphertext(cc_, c1);
		GPUct.rotate(i, true);

		// The used key is resident and fits the budget (other keys were evicted).
		ASSERT_TRUE(cc_->IsRotationKeyResident(i));
		ASSERT_LE(cc_->RotationKeyCacheResidentBytes(), one_key);

		ExpectRotationMatchesOpenFHE(c1, GPUct, i);
	}
}

// 3. Manual offload + reload round trip: a rotation computed with a freshly reloaded
// key must be bit-exact identical to the same rotation computed before the offload.
TEST_P(RotationKeyCacheTest, OffloadReloadRotationIsBitExact) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40);
	AddRotationKeys(cc_, true);

	auto c1 = MakePlaintextCiphertext();

	FIDESlib::CKKS::Ciphertext GPUct1 = MakeGpuCiphertext(cc_, c1);
	GPUct1.rotate(2, true); // lazily loads key 2
	ASSERT_TRUE(cc_->IsRotationKeyResident(2));
	FIDESlib::CKKS::RawCipherText res1;
	GPUct1.store(res1);

	cc_->OffloadRotationKeys(); // evict everything to host RAM
	ASSERT_FALSE(cc_->IsRotationKeyResident(2));
	ASSERT_EQ(cc_->RotationKeyCacheResidentBytes(), (size_t)0);

	FIDESlib::CKKS::Ciphertext GPUct2 = MakeGpuCiphertext(cc_, c1);
	GPUct2.rotate(2, true); // reloads key 2 from the snapshot
	ASSERT_TRUE(cc_->IsRotationKeyResident(2));
	FIDESlib::CKKS::RawCipherText res2;
	GPUct2.store(res2);

	ASSERT_EQ(res1.sub_0, res2.sub_0);
	ASSERT_EQ(res1.sub_1, res2.sub_1);
	ASSERT_EQ(res1.NoiseLevel, res2.NoiseLevel);
	ASSERT_EQ(res1.Noise, res2.Noise);
	ASSERT_EQ(res1.numRes, res2.numRes);
	ASSERT_EQ(res1.keyid, res2.keyid);
}

// 4. LRU eviction under a one-key budget: consecutive rotations evict the previously
// used key, and the resident-bytes accounting always reflects reality.
TEST_P(RotationKeyCacheTest, LRUEvictionBetweenRotations) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40);
	size_t one_key = AddRotationKeys(cc_, true);
	cc_->SetRotationKeyCache(one_key);

	auto c1 = MakePlaintextCiphertext();

	FIDESlib::CKKS::Ciphertext GPUct1 = MakeGpuCiphertext(cc_, c1);
	GPUct1.rotate(1, true);
	ASSERT_TRUE(cc_->IsRotationKeyResident(1));

	// Fresh ciphertext: rotations here must each be a rotation of the ORIGINAL plaintext.
	FIDESlib::CKKS::Ciphertext GPUct2 = MakeGpuCiphertext(cc_, c1);
	GPUct2.rotate(2, true);
	// key 1 was the LRU victim of loading key 2.
	ASSERT_FALSE(cc_->IsRotationKeyResident(1));
	ASSERT_TRUE(cc_->IsRotationKeyResident(2));
	ASSERT_LE(cc_->RotationKeyCacheResidentBytes(), one_key);

	ExpectRotationMatchesOpenFHE(c1, GPUct2, 2);
}

// 5. Pinned keys are never evicted, even when the budget is too small to hold them
// alongside the working set (documented transient overshoot instead of thrashing).
TEST_P(RotationKeyCacheTest, PinnedKeySurvivesEviction) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40);
	size_t one_key = AddRotationKeys(cc_, true);
	cc_->SetRotationKeyCache(one_key);
	cc_->PinRotationKey(1);

	auto c1 = MakePlaintextCiphertext();

	FIDESlib::CKKS::Ciphertext GPUct1 = MakeGpuCiphertext(cc_, c1);
	GPUct1.rotate(1, true); // loads (and keeps) the pinned key 1
	ASSERT_TRUE(cc_->IsRotationKeyResident(1));

	FIDESlib::CKKS::Ciphertext GPUct2 = MakeGpuCiphertext(cc_, c1);
	GPUct2.rotate(2, true); // key 2 loads; key 1 is pinned so it stays, no victim => overshoot
	ASSERT_TRUE(cc_->IsRotationKeyResident(1));

	FIDESlib::CKKS::Ciphertext GPUct3 = MakeGpuCiphertext(cc_, c1);
	GPUct3.rotate(3, true); // now key 2 is the LRU victim
	ASSERT_TRUE(cc_->IsRotationKeyResident(1));
	ASSERT_FALSE(cc_->IsRotationKeyResident(2));
	ASSERT_TRUE(cc_->IsRotationKeyResident(3));

	// Pinned key 1 + one working key: transient cap of 2 keys.
	ASSERT_LE(cc_->RotationKeyCacheResidentBytes(), 2 * one_key);

	FIDESlib::CKKS::Ciphertext GPUct4 = MakeGpuCiphertext(cc_, c1);
	GPUct4.rotate(4, true);
	ASSERT_TRUE(cc_->IsRotationKeyResident(1));
	ExpectRotationMatchesOpenFHE(c1, GPUct4, 4);
}

// 6. Hoisted rotation needs SEVERAL keys co-resident (they are fetched before the
// consuming kernels are enqueued); the eviction hold must prevent mid-op evictions and
// the results must still be correct with a tight budget.
TEST_P(RotationKeyCacheTest, HoistedRotateUnderTightBudget) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	cc_->SetRotationKeyCache(1ull << 40);
	size_t one_key = AddRotationKeys(cc_, true);
	cc_->SetRotationKeyCache(2 * one_key); // tighter than the 4-key hoisted set

	auto c1 = MakePlaintextCiphertext();
	auto raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	for (int hoisting_mode = 0; hoisting_mode < 2; ++hoisting_mode) {
		FIDESlib::CKKS::hoistRotateFused = hoisting_mode;

		FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
		FIDESlib::CKKS::Ciphertext GPUr1(cc_, raw1), GPUr2(cc_, raw1), GPUr3(cc_, raw1), GPUr4(cc_, raw1);
		GPUct.rotate_hoisted({1, 2, 3, 4}, {&GPUr1, &GPUr2, &GPUr3, &GPUr4}, false);

		ExpectRotationMatchesOpenFHE(c1, GPUr1, 1);
		ExpectRotationMatchesOpenFHE(c1, GPUr2, 2);
		ExpectRotationMatchesOpenFHE(c1, GPUr3, 3);
		ExpectRotationMatchesOpenFHE(c1, GPUr4, 4);
	}
	// After the op, the cache must shrink back to the budget on the next load
	// (here: just verify accounting sanity).
	ASSERT_LE(cc_->RotationKeyCacheResidentBytes(), 4 * one_key);
}

// 7. Legacy behavior must be untouched when no budget is set: keys load eagerly and
// stay resident.
TEST_P(RotationKeyCacheTest, DefaultUnlimitedKeepsKeysResident) {
	FIDESlib::CKKS::Context& cc_ = SetupGPUContext();

	ASSERT_EQ(cc_->GetRotationKeyCache(), SIZE_MAX);
	size_t one_key = AddRotationKeys(cc_, false); // eager
	(void)one_key;

	for (const int i : {1, 2, 3, 4}) {
		ASSERT_TRUE(cc_->IsRotationKeyResident(i));
	}

	auto c1 = MakePlaintextCiphertext();
	FIDESlib::CKKS::Ciphertext GPUct = MakeGpuCiphertext(cc_, c1);
	GPUct.rotate(3, true);
	ExpectRotationMatchesOpenFHE(c1, GPUct, 3);
}

INSTANTIATE_TEST_SUITE_P(RotationKeyCacheTests, RotationKeyCacheTest, testing::Values(tparams64_17_flex));

} // namespace FIDESlib::Testing
