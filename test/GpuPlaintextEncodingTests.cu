// GPU plaintext encoding (CryptoContextImpl::SetGPUPlaintextEncoding, on by default): the CPU only runs the IFFT
// and rounding, the GPU reduces modulo each prime and takes the NTT. The device limbs must be
// bit-identical to the ones uploaded from OpenFHE's own encoding, for every scaling technique,
// data type, packing, level and noise scale degree, and after cache eviction and reload.
#include <openfhe.h>
#undef duration
#include "ParametrizedTest.cuh"
#include "fideslib.hpp"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <cuda_runtime_api.h>

#include <random>

using namespace fideslib;

namespace {
struct DeviceImage {
    std::vector<std::vector<uint64_t>> rows;
    double noise = 0;
    int noiseLevel = 0;
    int slots = 0;
};

DeviceImage Download(CryptoContext<DCRTPoly>& cc, Plaintext& pt) {
    cc->LoadPlaintext(pt);
    auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(cc->GetDevicePlaintext(pt->gpu));
    DeviceImage image;
    gpu->c0.store(image.rows);
    cudaDeviceSynchronize();
    image.noise = gpu->NoiseFactor;
    image.noiseLevel = gpu->NoiseLevel;
    image.slots = gpu->slots;
    return image;
}

void ExpectSameImage(const DeviceImage& expected, const DeviceImage& actual) {
    ASSERT_EQ(expected.rows.size(), actual.rows.size());
    for (size_t i = 0; i < expected.rows.size(); ++i)
        ASSERT_EQ(expected.rows[i], actual.rows[i]) << "limb " << i;
    EXPECT_EQ(expected.noise, actual.noise);
    EXPECT_EQ(expected.noiseLevel, actual.noiseLevel);
    EXPECT_EQ(expected.slots, actual.slots);
}

std::vector<std::complex<double>> RandomValues(size_t n, double magnitude, bool complex, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-magnitude, magnitude);
    std::vector<std::complex<double>> values(n);
    for (auto& v : values)
        v = { dist(rng), complex ? dist(rng) : 0.0 };
    return values;
}

struct EncodingCase {
    ScalingTechnique technique;
    CKKSDataType datatype;
    uint32_t logN;
    uint32_t depth;
    uint32_t scaleBits;
    uint32_t batch; // context batch size
};

std::string CaseName(const testing::TestParamInfo<EncodingCase>& info) {
    const auto& c = info.param;
    std::string t = c.technique == FIXEDMANUAL ? "FixedManual" : c.technique == FIXEDAUTO ? "FixedAuto" : c.technique == FLEXIBLEAUTO ? "FlexAuto" : "FlexAutoExt";
    return t + (c.datatype == COMPLEX ? "Complex" : "Real") + "N" + std::to_string(c.logN) + "S" + std::to_string(c.scaleBits);
}

CryptoContext<DCRTPoly> MakeContext(const EncodingCase& c, KeyPair<DCRTPoly>& keys) {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetScalingTechnique(c.technique);
    params.SetCKKSDataType(c.datatype);
    params.SetMultiplicativeDepth(c.depth);
    params.SetScalingModSize(c.scaleBits);
    params.SetFirstModSize(60);
    params.SetBatchSize(c.batch);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << c.logN);
    auto cc = GenCryptoContext(params);
    for (const auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
        cc->Enable(feature);
    keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);
    return cc;
}

class GpuPlaintextEncoding : public testing::TestWithParam<EncodingCase> {};

TEST_P(GpuPlaintextEncoding, BitIdenticalToCpuEncoding) {
    const auto c = GetParam();
    KeyPair<DCRTPoly> keys;
    auto cc = MakeContext(c, keys);
    const uint32_t N = 1u << c.logN;
    const uint32_t maxLevel = c.depth - (c.technique == FLEXIBLEAUTOEXT ? 0 : 1);
    const bool complex = c.datatype == COMPLEX;

    int cases = 0, wide = 0;
    for (uint32_t slots : { N / 2, N / 8, 4u }) {
        for (uint32_t level : { 0u, 1u, maxLevel / 2, maxLevel }) {
            for (size_t degree : { 1, 2 }) {
                // Unit-range values, values whose scaled IFFT exceeds 64 bits (logApprox > 0),
                // and a vector shorter than the slot count.
                for (double magnitude : { 1.0, 300.0 }) {
                    for (size_t count : { static_cast<size_t>(slots), static_cast<size_t>(std::max(1u, slots / 2 - 1)) }) {
                        SCOPED_TRACE(testing::Message() << "slots=" << slots << " level=" << level << " degree=" << degree
                                                        << " magnitude=" << magnitude << " count=" << count);
                        const auto values = RandomValues(count, magnitude, complex, 1000 * level + 10 * degree + slots + count);
                        cc->SetGPUPlaintextEncoding(false);
                        auto reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
                        ASSERT_EQ(reference->coefficients, nullptr);
                        cc->SetGPUPlaintextEncoding(true);
                        auto deferred = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
                        ASSERT_NE(deferred->coefficients, nullptr);
                        wide += degree == 1 && !deferred->coefficients->factors.empty(); // scaled values wider than 64 bits
                        EXPECT_EQ(deferred->GetLevel(), reference->GetLevel());
                        EXPECT_EQ(deferred->GetSlots(), reference->GetSlots());
                        ExpectSameImage(Download(cc, reference), Download(cc, deferred));
                        // The OpenFHE object, once encoded on demand, matches the CPU encoding too.
                        EXPECT_TRUE(*deferred == *reference);
                        EXPECT_EQ(std::any_cast<const lbcrypto::Plaintext&>(deferred->cpu)->GetNoiseScaleDeg(),
                            std::any_cast<const lbcrypto::Plaintext&>(reference->cpu)->GetNoiseScaleDeg());
                        ++cases;
                    }
                }
            }
        }
    }
    EXPECT_EQ(cases, 3 * 4 * 2 * 2 * 2);
    if (c.scaleBits >= 59)
        EXPECT_GT(wide, 0);
}

TEST_P(GpuPlaintextEncoding, RealOverloadAndDefaultSlots) {
    const auto c = GetParam();
    KeyPair<DCRTPoly> keys;
    auto cc = MakeContext(c, keys);
    std::vector<double> values(c.batch);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(0.1 * i);
    cc->SetGPUPlaintextEncoding(false);
    auto reference = cc->MakeCKKSPackedPlaintext(values);
    cc->SetGPUPlaintextEncoding(true);
    auto deferred = cc->MakeCKKSPackedPlaintext(values);
    ASSERT_NE(deferred->coefficients, nullptr);
    ExpectSameImage(Download(cc, reference), Download(cc, deferred));
}

INSTANTIATE_TEST_SUITE_P(Fork,
    GpuPlaintextEncoding,
    testing::Values(EncodingCase{ FIXEDMANUAL, REAL, 12, 6, 50, 16 },
        EncodingCase{ FIXEDAUTO, REAL, 12, 6, 50, 16 },
        EncodingCase{ FLEXIBLEAUTO, REAL, 12, 6, 50, 16 },
        EncodingCase{ FLEXIBLEAUTOEXT, REAL, 12, 6, 50, 16 },
        EncodingCase{ FLEXIBLEAUTOEXT, COMPLEX, 12, 6, 50, 16 },
        EncodingCase{ FLEXIBLEAUTO, COMPLEX, 13, 8, 59, 32 },
        EncodingCase{ FIXEDMANUAL, REAL, 13, 8, 59, 32 },
        EncodingCase{ FLEXIBLEAUTOEXT, REAL, 16, 12, 59, 8192 }),
    CaseName);

TEST(GpuPlaintextEncodingMultiGpu, BitIdenticalAcrossTwoDevices) {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count < 2)
        GTEST_SKIP() << "needs two visible GPUs";
    CCParams<CryptoContextCKKSRNS> params;
    params.SetScalingTechnique(FLEXIBLEAUTOEXT);
    params.SetMultiplicativeDepth(10);
    params.SetScalingModSize(59);
    params.SetFirstModSize(60);
    params.SetBatchSize(1 << 12);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << 14);
    auto cc = GenCryptoContext(params);
    for (const auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
        cc->Enable(feature);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->SetDevices({ 0, 1 });
    cc->LoadContext(keys.publicKey);
    for (uint32_t slots : { 1u << 13, 1u << 10 }) {
        for (uint32_t level : { 0u, 1u, 6u }) {
            SCOPED_TRACE(testing::Message() << "slots=" << slots << " level=" << level);
            const auto values = RandomValues(slots, 1, false, slots + level);
            cc->SetGPUPlaintextEncoding(false);
            auto reference = cc->MakeCKKSPackedPlaintext(values, 1, level, nullptr, slots);
            cc->SetGPUPlaintextEncoding(true);
            auto deferred = cc->MakeCKKSPackedPlaintext(values, 1, level, nullptr, slots);
            ASSERT_NE(deferred->coefficients, nullptr);
            ExpectSameImage(Download(cc, reference), Download(cc, deferred));
            auto input = cc->MakeCKKSPackedPlaintext(RandomValues(slots, 1, false, 99), 1, level, nullptr, slots);
            auto ct = cc->Encrypt(keys.publicKey, input);
            ASSERT_EQ_CIPHERTEXT(cc->EvalMult(ct, reference), cc->EvalMult(ct, deferred));
        }
    }
}

CryptoContext<DCRTPoly> SmallContext(KeyPair<DCRTPoly>& keys) {
    return MakeContext(EncodingCase{ FLEXIBLEAUTO, REAL, 12, 6, 50, 16 }, keys);
}

TEST(GpuPlaintextEncodingApi, OnByDefault) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    EXPECT_TRUE(cc->GetGPUPlaintextEncoding());
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 1, 2, 3 });
    EXPECT_NE(pt->coefficients, nullptr);
    EXPECT_TRUE(pt->HasGPUEncoding());
    cc->SetGPUPlaintextEncoding(false);
    auto cpu = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 1, 2, 3 });
    EXPECT_EQ(cpu->coefficients, nullptr);
    EXPECT_FALSE(cpu->HasGPUEncoding());
}

// Code that swaps its own OpenFHE plaintext into `cpu` gets that plaintext uploaded, not the
// coefficients of the one it replaced, and EnsureCpuEncoding() leaves it alone.
TEST(GpuPlaintextEncodingApi, ReplacedCpuPlaintextIsUploaded) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    const std::vector<double> original{ 1, 2, 3, 4 }, replacement{ -0.5, 0.25, 0.75, 2 };
    cc->SetGPUPlaintextEncoding(false);
    auto reference = cc->MakeCKKSPackedPlaintext(replacement, 1, 1, nullptr, 16);
    cc->SetGPUPlaintextEncoding(true);

    auto pt = cc->MakeCKKSPackedPlaintext(original, 1, 1, nullptr, 16);
    ASSERT_TRUE(pt->HasGPUEncoding());
    pt->cpu = std::make_any<lbcrypto::Plaintext>(context->MakeCKKSPackedPlaintext(replacement, 1, 1, nullptr, 16));
    EXPECT_FALSE(pt->HasGPUEncoding());
    pt->EnsureCpuEncoding();
    EXPECT_TRUE(*pt == *reference);
    ExpectSameImage(Download(cc, reference), Download(cc, pt));
    EXPECT_EQ(pt->coefficients, nullptr);

    // Also after the plaintext was already on the device and got evicted.
    auto loaded = cc->MakeCKKSPackedPlaintext(original, 1, 1, nullptr, 16);
    cc->LoadPlaintext(loaded);
    cc->OffloadPlaintexts();
    loaded->cpu = std::make_any<lbcrypto::Plaintext>(context->MakeCKKSPackedPlaintext(replacement, 1, 1, nullptr, 16));
    ExpectSameImage(Download(cc, reference), Download(cc, loaded));
}

// The metadata OpenFHE sets on an encoded plaintext is there once EnsureCpuEncoding() ran.
TEST(GpuPlaintextEncodingApi, MetadataAfterCpuEncoding) {
    KeyPair<DCRTPoly> keys;
    auto cc = MakeContext(EncodingCase{ FLEXIBLEAUTOEXT, REAL, 12, 6, 50, 16 }, keys);
    for (uint32_t level : { 0u, 2u }) {
        for (size_t degree : { 1, 2 }) {
            SCOPED_TRACE(testing::Message() << "level=" << level << " degree=" << degree);
            const std::vector<double> values{ 0.5, -0.25, 0.125 };
            cc->SetGPUPlaintextEncoding(false);
            auto reference = cc->MakeCKKSPackedPlaintext(values, degree, level);
            cc->SetGPUPlaintextEncoding(true);
            auto pt = cc->MakeCKKSPackedPlaintext(values, degree, level);
            pt->EnsureCpuEncoding();
            const auto& a = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
            const auto& b = std::any_cast<const lbcrypto::Plaintext&>(reference->cpu);
            EXPECT_EQ(a->GetNoiseScaleDeg(), b->GetNoiseScaleDeg());
            EXPECT_EQ(a->GetScalingFactor(), b->GetScalingFactor());
            EXPECT_EQ(a->GetLevel(), b->GetLevel());
            EXPECT_EQ(a->GetSlots(), b->GetSlots());
            EXPECT_TRUE(*a == *b);
        }
    }
}

TEST(GpuPlaintextEncodingApi, ReloadAfterEvictionStaysIdentical) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    const auto values = RandomValues(16, 1, false, 7);
    cc->SetGPUPlaintextEncoding(false);
    auto reference = cc->MakeCKKSPackedPlaintext(values, 1, 2, nullptr, 16);
    cc->SetGPUPlaintextEncoding(true);
    auto deferred = cc->MakeCKKSPackedPlaintext(values, 1, 2, nullptr, 16);
    auto input = cc->MakeCKKSPackedPlaintext(RandomValues(16, 1, false, 8), 1, 2, nullptr, 16);
    auto ct = cc->Encrypt(keys.publicKey, input);
    const auto expectedImage = Download(cc, reference);
    auto expected = cc->EvalMult(ct, reference);
    cc->SetPlaintextCache(0);
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto actual = cc->EvalMult(ct, deferred);
        ASSERT_EQ_CIPHERTEXT(expected, actual);
        cc->OffloadPlaintexts();
        EXPECT_FALSE(deferred->loaded);
        EXPECT_EQ(cc->GetPlaintextCacheResidentBytes(), 0u);
        ExpectSameImage(expectedImage, Download(cc, deferred));
    }
}

TEST(GpuPlaintextEncodingApi, EncryptEncodesOnTheCpuOnDemand) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    cc->SetGPUPlaintextEncoding(true);
    const std::vector<double> values{ 0.5, -0.25, 0.125, 1.0 };
    auto pt = cc->MakeCKKSPackedPlaintext(values);
    ASSERT_NE(pt->coefficients, nullptr);
    const auto& raw = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
    EXPECT_FALSE(raw->IsEncoded());
    auto ct = cc->Encrypt(keys.publicKey, pt);
    EXPECT_TRUE(raw->IsEncoded());
    Plaintext out;
    cc->Decrypt(keys.secretKey, ct, &out);
    out->SetLength(values.size());
    const auto decoded = out->GetRealPackedValue();
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_NEAR(decoded[i], values[i], 1e-6);
    // The GPU still finishes from the coefficients, consistent with the CPU encoding.
    cc->SetGPUPlaintextEncoding(false);
    auto reference = cc->MakeCKKSPackedPlaintext(values);
    ExpectSameImage(Download(cc, reference), Download(cc, pt));
}

TEST(GpuPlaintextEncodingApi, DecryptIntoDeferredPlaintextDropsCoefficients) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    cc->SetGPUPlaintextEncoding(true);
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 1, 2, 3, 4 });
    auto input = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 4, 3, 2, 1 });
    auto ct = cc->Encrypt(keys.publicKey, input);
    cc->Decrypt(keys.secretKey, ct, &pt);
    EXPECT_EQ(pt->coefficients, nullptr);
    pt->SetLength(4);
    EXPECT_NEAR(pt->GetRealPackedValue()[0], 4, 1e-6);
}

TEST(GpuPlaintextEncodingApi, ErrorsMatchTheCpuPath) {
    KeyPair<DCRTPoly> keys;
    auto cc = SmallContext(keys);
    for (bool gpu : { false, true }) {
        cc->SetGPUPlaintextEncoding(gpu);
        SCOPED_TRACE(gpu ? "gpu" : "cpu");
        EXPECT_THROW(cc->MakeCKKSPackedPlaintext(std::vector<double>{}), lbcrypto::OpenFHEException);
        EXPECT_THROW(cc->MakeCKKSPackedPlaintext(std::vector<double>(8, 1.0), 1, 0, nullptr, 4), lbcrypto::OpenFHEException);
        EXPECT_THROW(cc->MakeCKKSPackedPlaintext(std::vector<double>(4, 1.0), 1, 100), lbcrypto::OpenFHEException);
        EXPECT_THROW(cc->MakeCKKSPackedPlaintext(std::vector<double>(4, 1e-20)), lbcrypto::OpenFHEException);
    }
}

TEST(GpuPlaintextEncodingApi, CpuOnlyContextsEncodeOnTheCpu) {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3);
    params.SetScalingModSize(50);
    params.SetBatchSize(8);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetRingDim(1 << 10);
    auto cc = GenCryptoContext(params);
    for (const auto feature : { PKE, KEYSWITCH, LEVELEDSHE })
        cc->Enable(feature);
    cc->SetDevices({});
    cc->SetGPUPlaintextEncoding(true);
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>{ 1, 2 });
    EXPECT_EQ(pt->coefficients, nullptr);
    EXPECT_TRUE(std::any_cast<const lbcrypto::Plaintext&>(pt->cpu)->IsEncoded());
}
} // namespace
