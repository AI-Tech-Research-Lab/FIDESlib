// Fourier-extension functional bootstrapping (OpenFHE FHECKKSRNS::EvalFEFuncBootstrap*): bit-exact
// CPU/GPU coverage.
//
// FeFuncPrimitives exercises the complex-coefficient GPU primitives the pipeline is built from
// (evalChebyshevSeries, evalPowers, evalPolyWithPrecomp) in isolation, on inputs in the same state
// as inside the real pipeline, against the vendored OpenFHE evaluators.

#include "FeFuncCoefficients.hpp"
#include "ParametrizedTest.cuh"

#include <CKKS/ApproxModEval.cuh>
#include <CKKS/Ciphertext.cuh>

#include <any>
#include <complex>
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

} // namespace FIDESlib::Testing
