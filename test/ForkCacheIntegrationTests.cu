// Verify that fork memory budgets compose with the upstream 2.2.0 arithmetic,
// compressed bootstrap operands, iterative/StC-first modes and result export.
#include <openfhe.h>
#undef duration
#include "ParametrizedTest.cuh"
#include "fideslib.hpp"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Plaintext.cuh"

using namespace fideslib;

namespace {
struct BootstrapCacheCase {
    bool stcFirst;
    uint32_t iterations;
    uint32_t budget;
    CKKSDataType datatype;
    SecretKeyDist secret;
};

class ForkBootstrapCache : public testing::TestWithParam<BootstrapCacheCase> {};

TEST_P(ForkBootstrapCache, MatchesCpuAfterRepeatedEviction) {
    const auto mode = GetParam();
    constexpr uint32_t depth = 25, slots = 8;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(mode.secret);
    params.SetCKKSDataType(mode.datatype);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetMultiplicativeDepth(depth);
    params.SetScalingModSize(50);
    params.SetBatchSize(slots);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << 12);
    params.SetPlaintextAutoload(true);
    params.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(params);
    for (const auto feature : { PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE })
        cc->Enable(feature);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapSetup({ mode.budget, mode.budget }, { 0, 0 }, slots, 0, true, mode.stcFirst);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    std::vector<std::complex<double>> values{ { 0.25, 0.125 }, { 0.5, -0.25 }, { 0.75, 0.1 }, { 1.0, 0.2 } };
    const uint32_t level = mode.stcFirst ? depth + 1 - (mode.budget + 2) : depth - 1;
    auto pt = cc->MakeCKKSPackedPlaintext(values, 1, level, nullptr, slots);
    auto input = cc->Encrypt(keys.publicKey, pt);
    auto expected = cc->EvalBootstrap(input, mode.iterations, 10);

    cc->SetBootstrapCache(0);
    cc->SetRotationKeyCache(0);
    cc->SetCiphertextCache(0);
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);
    EXPECT_EQ(cc->GetBootstrapCacheResidentBytes(), 0u);
    EXPECT_EQ(cc->GetRotationKeyCacheResidentBytes(), 0u);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto loaded = cc->GetBootstrapCacheLoadedBytes();
        input->Offload();
        auto actual = cc->EvalBootstrap(input, mode.iterations, 10);
        ASSERT_EQ_CIPHERTEXT(expected, actual);
        EXPECT_GT(cc->GetBootstrapCacheLoadedBytes(), loaded);
        actual->Offload();
        cc->UnloadCiphertext(actual);
        ASSERT_EQ_CIPHERTEXT(expected, actual);
        cc->OffloadBootstrapPrecomputation();
        EXPECT_EQ(cc->GetBootstrapCacheResidentBytes(), 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(Upstream220,
    ForkBootstrapCache,
    testing::Values(BootstrapCacheCase{ false, 1, 3, REAL, UNIFORM_TERNARY },
        BootstrapCacheCase{ false, 2, 3, REAL, UNIFORM_TERNARY },
        BootstrapCacheCase{ true, 1, 3, REAL, UNIFORM_TERNARY },
        BootstrapCacheCase{ true, 2, 3, REAL, UNIFORM_TERNARY },
        BootstrapCacheCase{ false, 1, 1, REAL, UNIFORM_TERNARY },
        BootstrapCacheCase{ false, 1, 3, COMPLEX, UNIFORM_TERNARY },
        BootstrapCacheCase{ false, 2, 3, REAL, SPARSE_ENCAPSULATED }));

TEST(ForkCacheIntegration, CompressedPlaintextsAndCiphertextsUnderZeroBudget) {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(50);
    params.SetBatchSize(8);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(128);
    params.SetPlaintextAutoload(true);
    params.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(params);
    for (const auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
        cc->Enable(feature);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    auto pa = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 1, 2, 3, 4 });
    auto pb = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 4, 3, 2, 1 });
    auto a = cc->Encrypt(keys.publicKey, pa);
    auto b = cc->Encrypt(keys.publicKey, pb);
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 0.5, 0.25, 0.75, 1 }, 1, 0, nullptr, 8);
    auto expected = cc->EvalAdd(cc->EvalMult(a, pt), b);
    cc->SetDevices({ 0 });
    cc->SetPlaintextCache(0);
    cc->SetCiphertextCache(0);
    cc->LoadContext(keys.publicKey);
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto actual = cc->EvalAdd(cc->EvalMult(a, pt), b);
        ASSERT_EQ_CIPHERTEXT(expected, actual);
        cc->OffloadPlaintexts();
        cc->OffloadCiphertexts();
        EXPECT_EQ(cc->GetPlaintextCacheResidentBytes(), 0u);
        EXPECT_EQ(cc->GetCiphertextCacheResidentBytes(), 0u);
        actual->Reload();
        ASSERT_EQ_CIPHERTEXT(expected, actual);
    }
}
} // namespace

TEST(ForkCacheIntegration, RepeatedCompressedPlaintextMultiplication) {
    for (uint32_t slots : { 512u, 1024u, 2048u, 4096u }) {
        SCOPED_TRACE(slots);
        CCParams<CryptoContextCKKSRNS> params;
        params.SetRingDim(16384);
        params.SetSecurityLevel(HEStd_NotSet);
        params.SetMultiplicativeDepth(8);
        params.SetScalingModSize(59);
        params.SetFirstModSize(60);
        params.SetBatchSize(slots);
        params.SetScalingTechnique(FLEXIBLEAUTO);
        params.SetPlaintextAutoload(true);
        params.SetCiphertextAutoload(true);
        auto cc = GenCryptoContext(params);
        for (auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
            cc->Enable(feature);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.25));
        auto one = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 1.0));
        auto input = cc->Encrypt(keys.publicKey, pt);
        auto expected = input->Clone();
        for (int i = 0; i < 3; ++i)
            expected = cc->EvalMult(expected, one);
        cc->SetDevices({ 0 });
        cc->LoadContext(keys.publicKey);
        auto actual = input->Clone();
        for (int i = 0; i < 3; ++i)
            actual = cc->EvalMult(actual, one);
        ASSERT_EQ_CIPHERTEXT(expected, actual);
    }
}

TEST(ForkCacheIntegration, CpuAddManyInPlaceDetachesClones) {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(128);
    params.SetBatchSize(8);
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(50);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    auto keys = cc->KeyGen();
    std::vector<double> values{ 0.25, -0.5, 0.75, 1.0 };
    auto pt = cc->MakeCKKSPackedPlaintext(values);
    auto input = cc->Encrypt(keys.publicKey, pt);
    std::vector<Ciphertext<DCRTPoly>> clones{ input->Clone(), input->Clone(), input->Clone() };
    auto alias = clones.front();
    cc->EvalAddManyInPlace(clones);
    for (auto result : { input, alias }) {
        Plaintext decoded;
        cc->Decrypt(keys.secretKey, result, &decoded);
        decoded->SetLength(values.size());
        const auto actual = decoded->GetRealPackedValue();
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_NEAR(actual[i], values[i] * (result == input ? 1 : 3), 1e-5);
    }
}

TEST(ForkCacheIntegration, DotProductWithCompressedPlaintextsAndAliasedOutput) {
    for (uint32_t slots : { 8u, 64u }) {
        SCOPED_TRACE(slots);
        CCParams<CryptoContextCKKSRNS> params;
        params.SetSecurityLevel(HEStd_NotSet);
        params.SetRingDim(128);
        params.SetBatchSize(slots);
        params.SetMultiplicativeDepth(4);
        params.SetScalingModSize(50);
        params.SetScalingTechnique(FLEXIBLEAUTO);
        auto cc = GenCryptoContext(params);
        for (auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
            cc->Enable(feature);
        auto keys = cc->KeyGen();
        std::vector<Plaintext> weights;
        std::vector<std::vector<double>> masks;
        std::vector<Ciphertext<DCRTPoly>> inputs;
        for (int k = 0; k < 3; ++k) {
            std::vector<double> values(slots), mask(slots);
            for (uint32_t i = 0; i < slots; ++i) {
                values[i] = (static_cast<double>(i) - slots / 2) / slots + 0.1 * k;
                mask[i] = (i + 1.0) / slots * (k + 1);
            }
            auto pt = cc->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, slots);
            inputs.push_back(cc->Encrypt(keys.publicKey, pt));
            masks.push_back(mask);
            weights.push_back(cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));
        }
        auto expected = cc->EvalMult(inputs[0], weights[0]);
        for (int k = 1; k < 3; ++k)
            cc->EvalAddInPlace(expected, cc->EvalMult(inputs[k], weights[k]));
        // Keep the plaintext operands independent between the CPU and GPU phases.
        for (int k = 0; k < 3; ++k)
            weights[k] = cc->MakeCKKSPackedPlaintext(masks[k], 1, 0, nullptr, slots);
        cc->SetDevices(devices);
        cc->LoadContext(keys.publicKey);
        std::vector<FIDESlib::CKKS::Ciphertext*> cts;
        std::vector<FIDESlib::CKKS::Plaintext*> pts;
        for (int k = 0; k < 3; ++k) {
            cc->LoadCiphertext(inputs[k]);
            cc->LoadPlaintext(weights[k]);
            cts.push_back(std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(inputs[k]->gpu)).get());
            pts.push_back(std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(cc->GetDevicePlaintext(weights[k]->gpu)).get());
        }
        for (int k = 0; k < 3; ++k) {
            ASSERT_EQ(cts[k]->NoiseLevel, 1);
            ASSERT_EQ(pts[k]->NoiseLevel, 1);
        }
        cts[0]->dotProductPt(cts.data(), pts.data(), 3, false);
        ASSERT_EQ_CIPHERTEXT(expected, inputs[0]);
    }
}

TEST(ForkCacheIntegration, SparseAuxiliaryContextReleasedWithApiOwner) {
    std::weak_ptr<FIDESlib::CKKS::ContextData> mainContext, sparseContext;
    for (int repeat = 0; repeat < 2; ++repeat) {
        {
            CCParams<CryptoContextCKKSRNS> params;
            params.SetSecurityLevel(HEStd_NotSet);
            params.SetRingDim(4096);
            params.SetBatchSize(8);
            params.SetMultiplicativeDepth(25);
            params.SetScalingModSize(59);
            params.SetFirstModSize(60);
            params.SetScalingTechnique(FLEXIBLEAUTO);
            params.SetSecretKeyDist(SPARSE_ENCAPSULATED);
            auto cc = GenCryptoContext(params);
            for (auto feature : { PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE })
                cc->Enable(feature);
            auto keys = cc->KeyGen();
            cc->EvalMultKeyGen(keys.secretKey);
            cc->EvalBootstrapSetup({ 3, 3 }, { 0, 0 }, 8);
            cc->EvalBootstrapKeyGen(keys.secretKey, 8);
            cc->SetDevices(devices);
            cc->LoadContext(keys.publicKey);
            auto& gpu = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
            mainContext = gpu;
            sparseContext = gpu->GetBootPrecomputation(8).sparse_context;
            ASSERT_FALSE(sparseContext.expired());
        }
        EXPECT_TRUE(mainContext.expired());
        EXPECT_TRUE(sparseContext.expired());
    }
}
