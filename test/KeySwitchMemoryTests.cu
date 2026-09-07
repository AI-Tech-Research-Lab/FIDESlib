// Repeated key switches must release their temporary CUDA pointer tables.
// Measure CUDA's USED bytes: the leaked allocations bypass FIDESlib's slab pool,
// and reserved VRAM can hide small leaks behind allocator reservation granularity.
#include <openfhe.h>
#undef duration
#include <fideslib.hpp>
#include "CKKS/Context.cuh"
#include "ParametrizedTest.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <any>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void CheckCuda(cudaError_t error) {
	if (error != cudaSuccess)
		throw std::runtime_error(cudaGetErrorString(error));
}

std::vector<uint64_t> UsedBytes(const fideslib::CryptoContext<fideslib::DCRTPoly>& cc) {
	cc->Synchronize();
	auto& gpu = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
	gpu->clearAuxilarPoly();
	cc->Synchronize();
	cc->TrimGPUMemoryPool();
	std::vector<uint64_t> bytes;
	for (int device : devices) {
		CheckCuda(cudaSetDevice(device));
		CheckCuda(cudaDeviceSynchronize());
		cudaMemPool_t pool;
		CheckCuda(cudaDeviceGetDefaultMemPool(&pool, device));
		uint64_t used = 0;
		CheckCuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
		bytes.push_back(used);
	}
	return bytes;
}

void CheckRepeatedOperation(bool bootstrap) {
	using namespace fideslib;
	if (FIDESlib::CKKS::GRAPH_CAPTURE)
		GTEST_SKIP() << "This regression exercises non-captured pointer-table ownership";

	constexpr int slots = 4096;
	constexpr int depth = 30;
	CCParams<CryptoContextCKKSRNS> params;
	params.SetSecurityLevel(HEStd_NotSet);
	params.SetRingDim(8192);
	params.SetMultiplicativeDepth(depth);
	params.SetScalingModSize(59);
	params.SetFirstModSize(60);
	params.SetNumLargeDigits(3);
	params.SetBatchSize(slots);
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetKeySwitchTechnique(HYBRID);
	params.SetSecretKeyDist(SPARSE_ENCAPSULATED);
	params.SetDevices(std::vector<int>(devices));
	auto cc = GenCryptoContext(params);
	for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE})
		cc->Enable(feature);
	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	if (bootstrap) {
		cc->EvalBootstrapSetup({4, 4}, {0, 0}, slots);
		cc->EvalBootstrapKeyGen(keys.secretKey, slots);
	}
	cc->LoadContext(keys.publicKey);
	std::vector<double> input(slots);
	for (int i = 0; i < slots; ++i)
		input[i] = 0.01 + 0.0001 * (i % 17);
	auto pt = cc->MakeCKKSPackedPlaintext(input, 1, bootstrap ? depth - 1 : 0, nullptr, slots);
	auto ct = cc->Encrypt(keys.publicKey, pt);
	auto operation = [&]() { return bootstrap ? cc->EvalBootstrap(ct, 2, 8) : cc->EvalMult(ct, ct); };
	auto batch = [&]() {
		// No per-operation synchronization: freeing on the wrong stream must not be
		// hidden by a device-wide fence between successive operations.
		for (int i = 0; i < (bootstrap ? 10 : 100); ++i) {
			auto result = operation();
			if (i == 0) {
				Plaintext decoded;
				cc->Decrypt(keys.secretKey, result, &decoded);
				decoded->SetLength(slots);
				const auto& values = decoded->GetRealPackedValue();
				ASSERT_EQ(values.size(), input.size());
				for (int j = 0; j < slots; ++j) {
					const double expected = bootstrap ? input[j] : input[j] * input[j];
					ASSERT_TRUE(std::isfinite(values[j]));
					ASSERT_NEAR(values[j], expected, bootstrap ? 1e-5 : 1e-8) << "slot " << j;
				}
			}
		}
	};
	batch();
	ASSERT_FALSE(::testing::Test::HasFatalFailure());
	const auto baseline = UsedBytes(cc);
	for (int round = 0; round < 3; ++round) {
		batch();
		ASSERT_FALSE(::testing::Test::HasFatalFailure());
		EXPECT_EQ(UsedBytes(cc), baseline) << "CUDA allocations grew after round " << round;
	}
}

} // namespace

TEST(KeySwitchMemory, RepeatedMultiplication) {
	CheckRepeatedOperation(false);
}

TEST(KeySwitchMemory, RepeatedIterativeBootstrap) {
	CheckRepeatedOperation(true);
}
