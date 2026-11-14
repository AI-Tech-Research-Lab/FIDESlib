// Fourier-extension functional bootstrapping (OpenFHE FHECKKSRNS::EvalFEFuncBootstrap*): bit-exact
// CPU/GPU coverage.
//
// FeFuncPrimitives exercises the complex-coefficient GPU primitives the pipeline is built from
// (evalChebyshevSeries, evalPowers, evalPolyWithPrecomp) in isolation, on inputs in the same state
// as inside the real pipeline, against the vendored OpenFHE evaluators.
//
// FeFuncBootstrap runs the whole pipeline through the fideslib api: computed on the CPU, then on
// the GPU after LoadContext, compared bit for bit, decrypted, and checked against the target
// function in the clear.

#include "FeFuncCoefficients.hpp"
#include "ParametrizedTest.cuh"

#include <CKKS/ApproxModEval.cuh>
#include <CKKS/Ciphertext.cuh>

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include <fideslib.hpp>

using namespace fideslib;

namespace FIDESlib::Testing {

namespace {

using Coeffs = std::vector<std::complex<double>>;
using LCiphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

const std::vector<ScalingTechnique> kTechniques = { FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO, FIXEDMANUAL };

// The three complex-exponential Chebyshev tables of FHECKKSRNS::EvalFEFuncBootstrapExp, one per
// secret-key distribution.
const std::vector<std::pair<std::string, Coeffs>>& ExpTables() {
    static const std::vector<std::pair<std::string, Coeffs>> tables = {
        { "coeff_exp_696_double_27", lbcrypto::FHECKKSRNS::coeff_exp_696_double_27 },
        { "coeff_exp_28_double_48", lbcrypto::FHECKKSRNS::coeff_exp_28_double_48 },
        { "coeff_exp_16_double_23", lbcrypto::FHECKKSRNS::coeff_exp_16_double_23 },
    };
    return tables;
}

// The Fourier series of the OpenFHE FE example (degrees 29, 34 and 44).
const std::vector<std::pair<std::string, Coeffs>>& Series() {
    static const std::vector<std::pair<std::string, Coeffs>> series = {
        { "exp", fe_func_coefficients::coeff_exp_2_double_29 },
        { "sigmoid", fe_func_coefficients::coeff_sigmoid_8_double_34 },
        { "gelu", fe_func_coefficients::coeff_gelu_8_double_44 },
    };
    return series;
}

// Same shape as MakeSmallContext in OpenFheCompatTests.cu, deep enough for a degree-48
// Chebyshev series or two squarings followed by a degree-44 power series.
CryptoContext<DCRTPoly> MakePrimitiveContext(ScalingTechnique tech) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(tech);
    parameters.SetMultiplicativeDepth(14);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& Lbcrypto(CryptoContext<DCRTPoly>& cc) {
    return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
}

// Wraps an OpenFHE ciphertext into a (not yet loaded) fideslib ciphertext.
Ciphertext<DCRTPoly> Wrap(CryptoContext<DCRTPoly>& cc, const LCiphertext& c) {
    Ciphertext<DCRTPoly> w = std::make_shared<CiphertextImpl<DCRTPoly>>(CryptoContext<DCRTPoly>(cc));
    w->cpu = std::make_any<LCiphertext>(c->Clone());
    return w;
}

// A loaded copy of `src` and its device ciphertext. The copy carries src's CPU metadata, and its
// device data is what ASSERT_EQ_CIPHERTEXT downloads and compares.
struct DeviceCopy {
    Ciphertext<DCRTPoly> api;
    std::shared_ptr<FIDESlib::CKKS::Ciphertext> dev;
};

DeviceCopy CopyToDevice(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& src) {
    cc->LoadCiphertext(src);
    DeviceCopy copy;
    copy.api = std::make_shared<CiphertextImpl<DCRTPoly>>(*src);
    copy.dev = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(copy.api->gpu));
    return copy;
}

LCiphertext FreshInput(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys) {
    std::vector<double> x = { -0.9, -0.62, -0.37, -0.05, 0.12, 0.33, 0.71, 0.88 };
    Plaintext pt = cc->MakeCKKSPackedPlaintext(x);
    auto fresh = cc->Encrypt(keys.publicKey, pt);
    return std::any_cast<LCiphertext>(fresh->cpu)->Clone();
}

// The state of the complex-exponential Chebyshev input inside the pipeline: after CoeffsToSlots and
// its ModReduce, i.e. noise degree 1 and not fresh (FIXEDMANUAL leaves the fresh ciphertext as is).
LCiphertext ChebyshevInput(CryptoContext<DCRTPoly>& cc, const LCiphertext& fresh, ScalingTechnique tech) {
    auto& lcc = Lbcrypto(cc);
    LCiphertext lct = fresh->Clone();
    if (tech != FIXEDMANUAL) {
        lct = lcc->EvalMult(lct, 1.0);
        while (lct->GetNoiseScaleDeg() > 1)
            lcc->GetScheme()->ModReduceInternalInPlace(lct, 1);
    }
    return lct;
}

// The state of the series input inside the pipeline: after double-angle squarings
// (EvalSquareInPlace + ModReduceInPlace), i.e. noise degree 2 under the AUTO techniques.
LCiphertext SeriesInput(CryptoContext<DCRTPoly>& cc, const LCiphertext& fresh) {
    auto& lcc = Lbcrypto(cc);
    LCiphertext lct = fresh->Clone();
    for (int i = 0; i < 2; ++i) {
        lcc->EvalSquareInPlace(lct);
        lcc->ModReduceInPlace(lct);
    }
    return lct;
}

} // namespace

// evalChebyshevSeries<complex> on the three complex-exponential tables, with the input in the
// state EvalFEFuncBootstrapExp hands to EvalChebyshevSeries.
TEST(FeFuncPrimitives, ComplexChebyshev) {
    for (auto tech : kTechniques) {
        auto cc = MakePrimitiveContext(tech);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        auto& lcc = Lbcrypto(cc);

        const LCiphertext input = ChebyshevInput(cc, FreshInput(cc, keys), tech);

        std::vector<LCiphertext> cpu;
        for (const auto& [name, table] : ExpTables())
            cpu.push_back(lcc->EvalChebyshevSeries(input, table, -1.0, 1.0));

        cc->SetDevices(devices);
        cc->LoadContext(keys.publicKey);
        auto wrapped = Wrap(cc, input);

        for (size_t t = 0; t < ExpTables().size(); ++t) {
            std::cout << "tech " << tech << ", " << ExpTables()[t].first << ": ";
            auto gpu = CopyToDevice(cc, wrapped);
            auto table = ExpTables()[t].second;
            FIDESlib::CKKS::evalChebyshevSeries(*gpu.dev, table, -1.0, 1.0);
            ASSERT_EQ_CIPHERTEXT(cpu[t], gpu.api);
        }
    }
}

// evalPowers + evalPolyWithPrecomp (fused = false) on the three Fourier series, against EvalPoly,
// with the input in the state left by the double-angle squarings.
TEST(FeFuncPrimitives, ComplexPolyPS) {
    for (auto tech : kTechniques) {
        auto cc = MakePrimitiveContext(tech);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        auto& lcc = Lbcrypto(cc);

        const LCiphertext input = SeriesInput(cc, FreshInput(cc, keys));

        std::vector<LCiphertext> cpu;
        for (const auto& [name, coeffs] : Series())
            cpu.push_back(lcc->EvalPoly(input, coeffs));

        cc->SetDevices(devices);
        cc->LoadContext(keys.publicKey);
        auto wrapped = Wrap(cc, input);

        for (size_t s = 0; s < Series().size(); ++s) {
            std::cout << "tech " << tech << ", " << Series()[s].first << ": ";
            const auto& coeffs = Series()[s].second;
            auto gpu = CopyToDevice(cc, wrapped);
            auto sp = FIDESlib::CKKS::evalPowers(*gpu.dev, coeffs, /*fused=*/false);
            auto res = FIDESlib::CKKS::evalPolyWithPrecomp(sp, coeffs, /*fused=*/false);
            gpu.dev->copy(res);
            ASSERT_EQ_CIPHERTEXT(cpu[s], gpu.api);
        }
    }
}

// One power set for the longest series, shared by all three series and evaluated twice each:
// equal to EvalPowers + EvalPolyWithPrecomp on the CPU, and the power set is never mutated.
TEST(FeFuncPrimitives, ComplexPolyShared) {
    const Coeffs& longest = fe_func_coefficients::coeff_gelu_8_double_44;
    for (auto tech : kTechniques) {
        auto cc = MakePrimitiveContext(tech);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        auto& lcc = Lbcrypto(cc);

        const LCiphertext input = SeriesInput(cc, FreshInput(cc, keys));

        auto cpuPowers = lcc->EvalPowers(input, longest);
        std::vector<LCiphertext> cpu;
        for (const auto& [name, coeffs] : Series())
            cpu.push_back(lcc->EvalPolyWithPrecomp(cpuPowers, coeffs));
        // The power-basis evaluation is the same computation as EvalPoly (guards the reference).
        ASSERT_EQ_CIPHERTEXT(cpu.back(), lcc->EvalPoly(input, longest));

        cc->SetDevices(devices);
        cc->LoadContext(keys.publicKey);
        auto wrapped = Wrap(cc, input);
        auto base = CopyToDevice(cc, wrapped);
        auto sp = FIDESlib::CKKS::evalPowers(*base.dev, longest, /*fused=*/false);

        for (size_t s = 0; s < Series().size(); ++s) {
            const auto& coeffs = Series()[s].second;
            std::vector<DeviceCopy> runs;
            for (int run = 0; run < 2; ++run) {
                auto gpu = CopyToDevice(cc, wrapped);
                gpu.dev->copy(FIDESlib::CKKS::evalPolyWithPrecomp(sp, coeffs, /*fused=*/false));
                runs.push_back(gpu);
            }
            std::cout << "tech " << tech << ", " << Series()[s].first << " (first): ";
            ASSERT_EQ_CIPHERTEXT(cpu[s], runs[0].api);
            std::cout << "tech " << tech << ", " << Series()[s].first << " (second): ";
            ASSERT_EQ_CIPHERTEXT(cpu[s], runs[1].api);
        }
    }
}

// Series below degree 5 take the linear power path; including gaps (zero intermediate
// coefficients), where the linear power set leaves the unneeded powers empty.
TEST(FeFuncPrimitives, ComplexPolyLinear) {
    const std::complex<double> c0(0.125, -0.03), c1(0.31, 0.27), c2(-0.22, 0.05), c3(0.07, -0.19), c4(0.04, 0.11);
    const std::vector<Coeffs> linear = {
        { c0, c1 },
        { c0, c1, c2 },
        { c0, 0.0, c2 },
        { c0, c1, c2, c3 },
        { c0, 0.0, 0.0, c3 },
        { c0, c1, c2, c3, c4 },
        { c0, 0.0, c2, 0.0, c4 },
        { c0, 0.0, 0.0, 0.0, c4 },
    };
    for (auto tech : kTechniques) {
        auto cc = MakePrimitiveContext(tech);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        auto& lcc = Lbcrypto(cc);

        const LCiphertext input = SeriesInput(cc, FreshInput(cc, keys));

        std::vector<LCiphertext> cpu;
        for (const auto& coeffs : linear)
            cpu.push_back(lcc->EvalPoly(input, coeffs));

        cc->SetDevices(devices);
        cc->LoadContext(keys.publicKey);
        auto wrapped = Wrap(cc, input);

        for (size_t s = 0; s < linear.size(); ++s) {
            std::cout << "tech " << tech << ", linear series " << s << ": ";
            auto gpu = CopyToDevice(cc, wrapped);
            auto sp = FIDESlib::CKKS::evalPowers(*gpu.dev, linear[s], /*fused=*/false);
            auto res = FIDESlib::CKKS::evalPolyWithPrecomp(sp, linear[s], /*fused=*/false);
            gpu.dev->copy(res);
            ASSERT_EQ_CIPHERTEXT(cpu[s], gpu.api);
        }
    }
}

//====================================================================
// Whole pipeline
//====================================================================

namespace {

double SigmoidFn(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
double ExpFn(double x) {
    return std::exp(x);
}
double GeluFn(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

// A target function f on [-radius, radius] and its Fourier series: on a message m in [-1/2, 1/2)
// the FE functional bootstrap evaluates g(m) = f(2 * radius * m).
struct FeTarget {
    const char* name;
    const Coeffs* coeffs;
    double radius;
    double (*f)(double);
};

const FeTarget kSigmoid{ "sigmoid", &fe_func_coefficients::coeff_sigmoid_8_double_34, 8.0, SigmoidFn };
const FeTarget kExp{ "exp", &fe_func_coefficients::coeff_exp_2_double_29, 2.0, ExpFn };
const FeTarget kGelu{ "gelu", &fe_func_coefficients::coeff_gelu_8_double_44, 8.0, GeluFn };

// 2 * Re(sum_j c_j exp(2 * pi * i * j * t)), t = m / 2: what the bootstrap computes, in the clear.
double EvalSeriesInClear(const Coeffs& coeffs, double m) {
    std::complex<double> acc = 0.0;
    for (size_t j = 0; j < coeffs.size(); ++j)
        acc += coeffs[j] * std::polar(1.0, M_PI * static_cast<double>(j) * m);
    return 2.0 * acc.real();
}

constexpr uint32_t kRingDim = 1 << 12;
// Bound on |decrypted - f| (the reference reaches ~1e-9 at these parameters).
constexpr double kFunctionTolerance = 1e-7;

struct FeCase {
    SecretKeyDist skd = SPARSE_TERNARY;
    ScalingTechnique tech = FLEXIBLEAUTO;
    uint32_t slots = 8; ///< 0 = full packing (N/2)
    std::vector<uint32_t> levelBudget = { 3, 2 };
    CKKSDataType dataType = REAL;
    /// Multiplicative depth on top of GetFEFBTDepth(levelBudget, series, skd).
    int extraDepth = 2;
    /// Encode the input this many levels above the depletion target (L0 - (lvlbDec + 2 + ext)).
    uint32_t aboveDepletion = 0;
};

std::string Describe(const FeCase& c) {
    return "skd " + std::to_string(c.skd) + ", tech " + std::to_string(c.tech) + ", slots " + std::to_string(c.slots) + ", budget {" +
        std::to_string(c.levelBudget[0]) + "," + std::to_string(c.levelBudget[1]) + "}, datatype " + std::to_string(c.dataType) +
        ", extra depth " + std::to_string(c.extraDepth) + ", above depletion " + std::to_string(c.aboveDepletion);
}

// A CPU-mode context with FE functional bootstrapping set up for `slots`, its keys, and an
// encrypted input -1/2 + i/slots (plus nonzero imaginary parts for COMPLEX) at the level the
// case asks for.
struct FeFixture {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
    uint32_t depth = 0;
    uint32_t slots = 0;
    std::vector<double> input;
    Ciphertext<DCRTPoly> ctxt;
};

FeFixture MakeFeFixture(const FeCase& c, const Coeffs& longest) {
    FeFixture fx;
    fx.slots = c.slots == 0 ? kRingDim / 2 : c.slots;
    fx.depth = static_cast<uint32_t>(static_cast<int>(fideslib::FHECKKSRNS::GetFEFBTDepth(c.levelBudget, longest, c.skd)) + c.extraDepth);

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(c.skd);
    parameters.SetScalingTechnique(c.tech);
    parameters.SetMultiplicativeDepth(fx.depth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetNumLargeDigits(3);
    parameters.SetBatchSize(fx.slots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(kRingDim);
    parameters.SetCKKSDataType(c.dataType);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    fx.cc = GenCryptoContext(parameters);
    fx.cc->Enable(PKE);
    fx.cc->Enable(KEYSWITCH);
    fx.cc->Enable(LEVELEDSHE);
    fx.cc->Enable(ADVANCEDSHE);
    fx.cc->Enable(FHE);

    fx.keys = fx.cc->KeyGen();
    fx.cc->EvalMultKeyGen(fx.keys.secretKey);
    fx.cc->EvalFEFuncBootstrapSetup(c.levelBudget, { 0, 0 }, fx.slots);
    fx.cc->EvalBootstrapKeyGen(fx.keys.secretKey, fx.slots);

    fx.input.resize(fx.slots);
    for (uint32_t i = 0; i < fx.slots; ++i)
        fx.input[i] = -0.5 + static_cast<double>(i) / fx.slots;

    const uint32_t level = fx.depth - (c.levelBudget[1] + 1) - c.aboveDepletion;
    Plaintext pt;
    if (c.dataType == COMPLEX) {
        std::vector<std::complex<double>> z(fx.slots);
        for (uint32_t i = 0; i < fx.slots; ++i)
            z[i] = { fx.input[i], 0.3 * std::sin(1.0 + i) };
        pt = fx.cc->MakeCKKSPackedPlaintext(z, 1, level, nullptr, fx.slots);
    } else {
        pt = fx.cc->MakeCKKSPackedPlaintext(fx.input, 1, level, nullptr, fx.slots);
    }
    fx.ctxt = fx.cc->Encrypt(fx.keys.publicKey, pt);
    return fx;
}

void LoadFixture(FeFixture& fx) {
    fx.cc->SetDevices(devices);
    fx.cc->LoadContext(fx.keys.publicKey);
}

// Decrypts both results, checks they agree, and checks the GPU one against `expected`: the real
// parts within kFunctionTolerance and, the output being real, the imaginary parts near zero.
void ExpectDecryptsTo(FeFixture& fx, Ciphertext<DCRTPoly>& cpu, Ciphertext<DCRTPoly>& gpu, const std::vector<double>& expected, CKKSDataType dataType) {
    Plaintext r1;
    fx.cc->Decrypt(fx.keys.secretKey, cpu, &r1);
    r1->SetLength(fx.slots);
    Plaintext r2;
    fx.cc->Decrypt(fx.keys.secretKey, gpu, &r2);
    r2->SetLength(fx.slots);

    if (dataType == REAL) {
        ASSERT_ERROR_OK(r1, r2);
    }
    const auto v1 = r1->GetCKKSPackedValue();
    const auto v2 = r2->GetCKKSPackedValue();
    ASSERT_EQ(v1.size(), expected.size());
    ASSERT_EQ(v2.size(), expected.size());
    double maxErr = 0.0, maxImag = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        maxErr = std::max({ maxErr, std::abs(v1[i].real() - expected[i]), std::abs(v2[i].real() - expected[i]) });
        maxImag = std::max({ maxImag, std::abs(v1[i].imag()), std::abs(v2[i].imag()) });
    }
    std::cout << "max |decrypted - f| = " << maxErr << ", max |imag| = " << maxImag << std::endl;
    EXPECT_LT(maxErr, kFunctionTolerance);
    EXPECT_LT(maxImag, kFunctionTolerance);
}

std::vector<double> Expected(const FeFixture& fx, const FeTarget& t) {
    std::vector<double> e(fx.input.size());
    for (size_t i = 0; i < e.size(); ++i)
        e[i] = t.f(2.0 * t.radius * fx.input[i]);
    return e;
}

// EvalFEFuncBootstrap on the CPU and on the GPU: bit-exact, and decrypting to f.
void RunFeFuncCase(const FeCase& c, const FeTarget& target) {
    SCOPED_TRACE(Describe(c) + ", " + target.name);
    std::cout << Describe(c) << ", " << target.name << std::endl;
    auto fx = MakeFeFixture(c, *target.coeffs);

    auto cpu = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *target.coeffs);

    LoadFixture(fx);
    auto gpu = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *target.coeffs);

    ASSERT_EQ_CIPHERTEXT(cpu, gpu);
    ExpectDecryptsTo(fx, cpu, gpu, Expected(fx, target), c.dataType);
}

} // namespace

// Case 1: the reference configuration (SPARSE_TERNARY, FLEXIBLEAUTO, 8 slots, {3, 2}).
TEST(FeFuncBootstrap, SparseTernarySparsePacking) {
    RunFeFuncCase(FeCase{}, kSigmoid);
}

// Case 2: level budget {1, 1}, i.e. the EvalLinearTransform branch of both transforms.
TEST(FeFuncBootstrap, SparseTernaryLinearTransform) {
    FeCase c;
    c.levelBudget = { 1, 1 };
    RunFeFuncCase(c, kSigmoid);
}

// Case 3: full packing (N/2 slots): no sparse fold and no PartialSum.
TEST(FeFuncBootstrap, SparseTernaryFullPacking) {
    FeCase c;
    c.slots = 0;
    RunFeFuncCase(c, kSigmoid);
}

// Case 4: UNIFORM_TERNARY (K = 696 table, 9 double-angle squarings).
TEST(FeFuncBootstrap, UniformTernary) {
    for (uint32_t slots : { 0u, 8u }) {
        FeCase c;
        c.skd = UNIFORM_TERNARY;
        c.slots = slots;
        RunFeFuncCase(c, kSigmoid);
    }
}

// Case 5: SPARSE_ENCAPSULATED (sparse-secret key switching around the raise; at a 60-bit first
// modulus the Hamming-weight-32 secret and the K = 16 table).
TEST(FeFuncBootstrap, SparseEncapsulated) {
    for (uint32_t slots : { 0u, 8u }) {
        FeCase c;
        c.skd = SPARSE_ENCAPSULATED;
        c.slots = slots;
        RunFeFuncCase(c, kSigmoid);
    }
}

// Case 6: the other three scaling techniques, sparse ({3, 2} and {1, 1}) and fully packed.
static void RunScalingTechniqueCases(ScalingTechnique tech) {
    for (const auto& [slots, budget] : std::vector<std::pair<uint32_t, std::vector<uint32_t>>>{ { 8, { 3, 2 } }, { 8, { 1, 1 } }, { 0, { 3, 2 } } }) {
        FeCase c;
        c.tech = tech;
        c.slots = slots;
        c.levelBudget = budget;
        RunFeFuncCase(c, kSigmoid);
    }
}

TEST(FeFuncBootstrap, FixedManual) {
    RunScalingTechniqueCases(FIXEDMANUAL);
}

TEST(FeFuncBootstrap, FixedAuto) {
    RunScalingTechniqueCases(FIXEDAUTO);
}

TEST(FeFuncBootstrap, FlexibleAutoExt) {
    RunScalingTechniqueCases(FLEXIBLEAUTOEXT);
}

// Case 7: the other two series of the OpenFHE example (degrees 29 and 44).
TEST(FeFuncBootstrap, ExpAndGelu) {
    FeCase c;
    c.slots = 0;
    RunFeFuncCase(c, kExp);
    RunFeFuncCase(c, kGelu);
}

// Case 8: the input encoded above the depletion target, so the depletion correction runs (the
// FLEXIBLEAUTO scale-correcting multiply, the FIXEDAUTO tower drop).
TEST(FeFuncBootstrap, InputAboveDepletionLevel) {
    for (auto tech : { FLEXIBLEAUTO, FIXEDAUTO }) {
        for (uint32_t above : { 1u, 3u }) {
            FeCase c;
            c.tech = tech;
            c.aboveDepletion = above;
            RunFeFuncCase(c, kSigmoid);
        }
    }
}

// Case 9: CKKSDataType COMPLEX with nonzero imaginary parts, which the bootstrap discards (the
// sparse fold is skipped for COMPLEX); the output is real.
TEST(FeFuncBootstrap, ComplexDataType) {
    FeCase full;
    full.dataType = COMPLEX;
    full.slots = 0;
    RunFeFuncCase(full, kSigmoid);

    FeCase sparse;
    sparse.dataType = COMPLEX;
    sparse.levelBudget = { 1, 1 };
    RunFeFuncCase(sparse, kSigmoid);
}


namespace {

// Short series (degree < 5: the linear power path), with zero intermediate coefficients.
const Coeffs kDeg1 = { { 0.1, 0.0 }, { 0.3, 0.05 } };
const Coeffs kDeg2 = { { 0.1, 0.0 }, { 0.2, 0.05 }, { -0.15, 0.1 } };
const Coeffs kDeg3Gap = { { 0.1, 0.0 }, { 0.2, 0.05 }, { 0.0, 0.0 }, { -0.15, 0.1 } };
const Coeffs kDeg4Gap = { { 0.05, 0.0 }, { 0.2, 0.05 }, { 0.1, 0.0 }, { 0.0, 0.0 }, { -0.15, 0.1 } };

std::vector<double> ExpectedSeries(const FeFixture& fx, const Coeffs& coeffs) {
    std::vector<double> e(fx.input.size());
    for (size_t i = 0; i < e.size(); ++i)
        e[i] = EvalSeriesInClear(coeffs, fx.input[i]);
    return e;
}

double MaxErrorTo(FeFixture& fx, Ciphertext<DCRTPoly>& ct, const std::vector<double>& expected) {
    Plaintext pt;
    fx.cc->Decrypt(fx.keys.secretKey, ct, &pt);
    pt->SetLength(fx.slots);
    const auto v = pt->GetCKKSPackedValue();
    double m = 0.0;
    for (size_t i = 0; i < expected.size(); ++i)
        m = std::max(m, std::abs(v[i].real() - expected[i]));
    return m;
}

} // namespace

// Case 10: one EvalFEFuncBootstrapPrecompute (for the longest series) shared by three functions,
// each evaluated twice: bit-exact against the CPU, the second evaluation identical to the first
// (the powers are never mutated), and the same as the single-function EvalFEFuncBootstrap.
TEST(FeFuncBootstrap, SharedPrecomputation) {
    FeCase c;
    c.slots = 0;
    const std::vector<FeTarget> targets = { kExp, kSigmoid, kGelu };
    auto fx = MakeFeFixture(c, *kGelu.coeffs);

    auto cpuPowers = fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, *kGelu.coeffs);
    std::vector<Ciphertext<DCRTPoly>> cpu;
    for (const auto& t : targets)
        cpu.push_back(fx.cc->EvalFEFuncBootstrapWithPrecomp(cpuPowers, *t.coeffs));
    auto cpuDirect = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *kGelu.coeffs);
    std::cout << "CPU precomputed gelu == CPU direct gelu: ";
    ASSERT_EQ_CIPHERTEXT(cpuDirect, cpu.back());

    LoadFixture(fx);
    // Powers built on the CPU are not converted implicitly once the context is loaded.
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(cpuPowers, *kSigmoid.coeffs), std::exception);

    auto gpuPowers = fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, *kGelu.coeffs);
    for (size_t t = 0; t < targets.size(); ++t) {
        for (int run = 0; run < 2; ++run) {
            SCOPED_TRACE(std::string(targets[t].name) + " run " + std::to_string(run));
            std::cout << targets[t].name << " run " << run << ": ";
            auto gpu = fx.cc->EvalFEFuncBootstrapWithPrecomp(gpuPowers, *targets[t].coeffs);
            ASSERT_EQ_CIPHERTEXT(cpu[t], gpu);
            ExpectDecryptsTo(fx, cpu[t], gpu, Expected(fx, targets[t]), REAL);
        }
    }
    auto gpuDirect = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *kGelu.coeffs);
    std::cout << "GPU direct gelu: ";
    ASSERT_EQ_CIPHERTEXT(cpuDirect, gpuDirect);
}

// Case 11: series below degree 5 (the linear power path) and series with trailing zeros, through
// EvalFEFuncBootstrap: the same outcome on both backends. A padded Paterson-Stockmeyer series is
// evaluated trimmed (same ciphertext as the unpadded one); a padded linear series is rejected.
TEST(FeFuncBootstrap, ShortAndPaddedSeries) {
    const std::vector<const Coeffs*> shortSeries = { &kDeg1, &kDeg2, &kDeg3Gap, &kDeg4Gap };
    Coeffs paddedSigmoid = *kSigmoid.coeffs;
    paddedSigmoid.resize(paddedSigmoid.size() + 3, 0.0);
    Coeffs paddedShort = kDeg2;
    paddedShort.push_back(0.0);

    auto fx = MakeFeFixture(FeCase{}, *kSigmoid.coeffs);

    std::vector<Ciphertext<DCRTPoly>> cpu;
    for (const auto* series : shortSeries)
        cpu.push_back(fx.cc->EvalFEFuncBootstrap(fx.ctxt, *series));
    auto cpuSigmoid = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *kSigmoid.coeffs);
    auto cpuPadded = fx.cc->EvalFEFuncBootstrap(fx.ctxt, paddedSigmoid);
    std::cout << "CPU padded sigmoid == sigmoid: ";
    ASSERT_EQ_CIPHERTEXT(cpuSigmoid, cpuPadded);
    // Rejected up front on both backends. (The reference alone does not reject it: the linear
    // EvalPowers leaves the top power empty and then dereferences it, a segmentation fault.)
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(fx.ctxt, paddedShort), std::invalid_argument);

    LoadFixture(fx);

    for (size_t s = 0; s < shortSeries.size(); ++s) {
        SCOPED_TRACE("short series of degree " + std::to_string(s + 1));
        std::cout << "short series " << s << ": ";
        auto gpu = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *shortSeries[s]);
        ASSERT_EQ_CIPHERTEXT(cpu[s], gpu);
        ExpectDecryptsTo(fx, cpu[s], gpu, ExpectedSeries(fx, *shortSeries[s]), REAL);
    }
    std::cout << "GPU padded sigmoid: ";
    auto gpuPadded = fx.cc->EvalFEFuncBootstrap(fx.ctxt, paddedSigmoid);
    ASSERT_EQ_CIPHERTEXT(cpuSigmoid, gpuPadded);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(fx.ctxt, paddedShort), std::invalid_argument);
}

namespace {

// Runs one case at its minimum workable depth (bit-exact and correct) and one level below it,
// where the GPU must reject it with a clean exception before touching anything, and the CPU
// reference indeed cannot produce it (it throws, or returns a ciphertext that does not decrypt
// to the series).
void RunLevelBoundary(FeCase c, const Coeffs& series, int minExtraDepth) {
    for (int extra : { minExtraDepth, minExtraDepth - 1 }) {
        c.extraDepth = extra;
        const bool atMinimum = extra == minExtraDepth;
        SCOPED_TRACE(Describe(c) + (atMinimum ? " (minimum)" : " (minimum - 1)"));
        std::cout << Describe(c) << (atMinimum ? " (minimum)" : " (minimum - 1)") << std::endl;
        auto fx = MakeFeFixture(c, series);
        const auto expected = ExpectedSeries(fx, series);

        Ciphertext<DCRTPoly> cpu;
        bool cpuThrew = false;
        try {
            cpu = fx.cc->EvalFEFuncBootstrap(fx.ctxt, series);
        } catch (const std::exception&) {
            cpuThrew = true;
        }

        LoadFixture(fx);
        if (atMinimum) {
            ASSERT_FALSE(cpuThrew);
            auto gpu = fx.cc->EvalFEFuncBootstrap(fx.ctxt, series);
            ASSERT_EQ_CIPHERTEXT(cpu, gpu);
            ExpectDecryptsTo(fx, cpu, gpu, expected, REAL);
        } else {
            EXPECT_TRUE(cpuThrew || MaxErrorTo(fx, cpu, expected) > 1e-3) << "the reference still works one level below the minimum";
            EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(fx.ctxt, series), std::invalid_argument);
        }
    }
}

} // namespace

// Case 12: the level-sufficiency check. GetFEFBTDepth counts the SlotsToCoeffs levels, which are
// spent before the modulus raise, so the minimum workable depth is levelBudget[1] below it for a
// Paterson-Stockmeyer series; GetFEFBTDepth undercounts the linear path (degree < 5) by one level,
// so for a short series the minimum is levelBudget[1] - 1 below it.
TEST(FeFuncBootstrap, LevelBoundary) {
    for (auto tech : kTechniques) {
        FeCase c;
        c.tech = tech;
        RunLevelBoundary(c, *kSigmoid.coeffs, -static_cast<int>(c.levelBudget[1]));
        RunLevelBoundary(c, kDeg2, -static_cast<int>(c.levelBudget[1]) + 1);
    }
    FeCase lt;
    lt.levelBudget = { 1, 1 };
    RunLevelBoundary(lt, *kSigmoid.coeffs, -1);
}

// Case 13: mode guards. A slot set up for FE functional bootstrapping cannot be bootstrapped
// normally and vice versa (the two fold different K into the CoeffsToSlots matrix), an
// unconfigured slot is rejected, and a conflicting second setup or a setup after LoadContext
// throws; on both backends.
TEST(FeFuncBootstrap, ModeGuards) {
    auto fx = MakeFeFixture(FeCase{}, *kSigmoid.coeffs);
    const uint32_t level = fx.depth - 3;
    std::vector<double> four = { -0.25, -0.1, 0.1, 0.25 };
    Plaintext pt4 = fx.cc->MakeCKKSPackedPlaintext(four, 1, level, nullptr, 4);
    auto ct4 = fx.cc->Encrypt(fx.keys.publicKey, pt4);

    EXPECT_THROW(fx.cc->EvalBootstrap(fx.ctxt), std::exception);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(ct4, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrapPrecompute(ct4, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(fx.cc->EvalBootstrapSetup({ 3, 2 }, { 0, 0 }, fx.slots), std::exception);
    EXPECT_THROW(fx.cc->EvalBootstrapSetup({ 3, 2 }, { 0, 0 }, fx.slots, 0, true, /*btsfirstboot=*/true), std::exception);
    // Re-running the same FE setup is not a conflict.
    EXPECT_NO_THROW(fx.cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, fx.slots));

    LoadFixture(fx);
    EXPECT_THROW(fx.cc->EvalBootstrap(fx.ctxt), std::exception);
    EXPECT_THROW(fx.cc->EvalBootstrapStCFirst(fx.ctxt), std::exception);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(ct4, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrapPrecompute(ct4, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, 16), std::exception);

    // A slot set up for the standard bootstrap.
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetMultiplicativeDepth(fx.depth);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);
    parameters.SetNumLargeDigits(3);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(kRingDim);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalBootstrapSetup({ 3, 2 }, { 0, 0 }, 8);
    cc->EvalBootstrapKeyGen(keys.secretKey, 8);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, 8), std::exception);
    Plaintext pt = cc->MakeCKKSPackedPlaintext(fx.input, 1, level, nullptr, 8);
    auto ct = cc->Encrypt(keys.publicKey, pt);
    EXPECT_THROW(cc->EvalFEFuncBootstrap(ct, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapPrecompute(ct, *kSigmoid.coeffs), std::exception);
    // Release the first GPU context before loading this one.
    fx = FeFixture{};
    cc->SetDevices(devices);
    cc->LoadContext(keys.publicKey);
    EXPECT_THROW(cc->EvalFEFuncBootstrap(ct, *kSigmoid.coeffs), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapPrecompute(ct, *kSigmoid.coeffs), std::exception);
}

// Case 13: setup arguments OpenFHE would index unchecked or only reject at the first evaluation.
TEST(FeFuncBootstrap, SetupValidation) {
    auto make = [](uint32_t scalingModSize, uint32_t firstModSize) {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetSecretKeyDist(SPARSE_TERNARY);
        parameters.SetMultiplicativeDepth(20);
        parameters.SetScalingModSize(scalingModSize);
        parameters.SetFirstModSize(firstModSize);
        parameters.SetBatchSize(8);
        parameters.SetSecurityLevel(HEStd_NotSet);
        parameters.SetRingDim(kRingDim);
        CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        return cc;
    };

    auto wrongFirstMod = make(50, 60);
    EXPECT_THROW(wrongFirstMod->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, 8), std::exception);

    auto cc = make(59, 60);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3 }, { 0, 0 }, 8), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3, 2, 1 }, { 0, 0 }, 8), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0 }, 8), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, 6), std::exception);
    EXPECT_THROW(cc->EvalFEFuncBootstrapSetup({ 3, 2 }, { 0, 0 }, kRingDim), std::exception);
    EXPECT_THROW(fideslib::FHECKKSRNS::GetFEFBTDepth({ 3 }, *kSigmoid.coeffs, SPARSE_TERNARY), std::exception);
    EXPECT_THROW(fideslib::FHECKKSRNS::GetFEFBTDepth({ 3, 2 }, Coeffs{}, SPARSE_TERNARY), std::exception);
    // The rejected setups left the slot unconfigured.
    EXPECT_NO_THROW(cc->EvalBootstrapSetup({ 3, 2 }, { 0, 0 }, 8));

    // GetFEFBTDepth forwards to OpenFHE with the matching distribution (table 2.2 of the plan).
    EXPECT_EQ(fideslib::FHECKKSRNS::GetFEFBTDepth({ 3, 2 }, *kSigmoid.coeffs, UNIFORM_TERNARY), 26u);
    EXPECT_EQ(fideslib::FHECKKSRNS::GetFEFBTDepth({ 3, 2 }, *kSigmoid.coeffs, SPARSE_TERNARY), 21u);
    EXPECT_EQ(fideslib::FHECKKSRNS::GetFEFBTDepth({ 3, 2 }, *kSigmoid.coeffs, SPARSE_ENCAPSULATED), 21u);
}

// Case 13: coefficient and power-handle errors, on both backends, and the input ciphertext stays
// usable after an evaluation threw mid-way.
TEST(FeFuncBootstrap, EvaluationErrors) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Coeffs withNan = *kSigmoid.coeffs;
    withNan[7] = { 0.0, nan };
    Coeffs withInf = *kSigmoid.coeffs;
    withInf[3] = { inf, 0.0 };
    const std::vector<Coeffs> invalid = { {}, { { 0.5, 0.0 } }, { { 0.5, 0.0 }, { 0.0, 0.0 } }, withNan, withInf };

    auto fx = MakeFeFixture(FeCase{}, *kGelu.coeffs);
    const auto shape = lbcrypto::ComputeDegreesPS(lbcrypto::Degree(*kGelu.coeffs));
    const uint32_t capacity = shape[0] * ((1u << shape[1]) - 1);
    // Degrees just below and at the capacity of the gelu power set.
    auto series = [](uint32_t degree) {
        Coeffs c(degree + 1);
        for (uint32_t j = 0; j <= degree; ++j)
            c[j] = { 0.004 * std::sin(j + 1.0), 0.004 * std::cos(2.0 * j) };
        return c;
    };
    const Coeffs belowCapacity = series(capacity - 1);
    const Coeffs atCapacity = series(capacity);
    // A series the context does not have the levels for: depth 11 against the 9 left over by the
    // series of the fixture (gelu, 6) and its 2 spare levels.
    const Coeffs tooDeep = series(2000);

    // Powers of another context.
    auto other = MakeFeFixture(FeCase{}, *kGelu.coeffs);

    auto checkBackend = [&](SeriesPowers& powers, const char* backend) {
        SCOPED_TRACE(backend);
        for (const auto& coeffs : invalid) {
            EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(fx.ctxt, coeffs), std::exception);
            EXPECT_THROW(fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, coeffs), std::exception);
            EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(powers, coeffs), std::exception);
        }
        EXPECT_THROW(fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, kDeg4Gap), std::exception);
        EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(nullptr, *kSigmoid.coeffs), std::exception);
        EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(std::make_shared<SeriesPowersImpl>(), *kSigmoid.coeffs), std::exception);
        EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(powers, atCapacity), std::exception);
        auto foreign = std::make_shared<SeriesPowersImpl>();
        foreign->exponential = other.ctxt;
        foreign->cpu = powers->cpu;
        foreign->gpu = powers->gpu;
        foreign->loaded = powers->loaded;
        EXPECT_THROW(fx.cc->EvalFEFuncBootstrapWithPrecomp(foreign, *kSigmoid.coeffs), std::exception);
    };

    auto cpuPowers = fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, *kGelu.coeffs);
    auto cpuBelow = fx.cc->EvalFEFuncBootstrapWithPrecomp(cpuPowers, belowCapacity);
    auto cpuSigmoid = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *kSigmoid.coeffs);
    checkBackend(cpuPowers, "CPU");

    LoadFixture(fx);
    auto gpuPowers = fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, *kGelu.coeffs);
    checkBackend(gpuPowers, "GPU");
    std::cout << "GPU series at capacity - 1: ";
    auto gpuBelow = fx.cc->EvalFEFuncBootstrapWithPrecomp(gpuPowers, belowCapacity);
    ASSERT_EQ_CIPHERTEXT(cpuBelow, gpuBelow);

    // The level check of the GPU core throws (after the api cloned the input, before any work);
    // the input is intact. Precompute validates its series against the levels the same way.
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrap(fx.ctxt, tooDeep), std::invalid_argument);
    EXPECT_THROW(fx.cc->EvalFEFuncBootstrapPrecompute(fx.ctxt, tooDeep), std::invalid_argument);
    std::cout << "GPU sigmoid after a rejected evaluation: ";
    auto gpuSigmoid = fx.cc->EvalFEFuncBootstrap(fx.ctxt, *kSigmoid.coeffs);
    ASSERT_EQ_CIPHERTEXT(cpuSigmoid, gpuSigmoid);
    ExpectDecryptsTo(fx, cpuSigmoid, gpuSigmoid, Expected(fx, kSigmoid), REAL);
}

} // namespace FIDESlib::Testing
