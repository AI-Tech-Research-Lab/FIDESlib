//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "forwardDefs.cuh"
#include "pke/openfhe.h"

#include <complex>
#include <vector>

namespace FIDESlib::CKKS {
struct SeriesPowers;

void BootstrapCPUraise(Ciphertext& ctxt,
    const int slots,
    std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
    const bool prescaled);

void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void Bootstrap(Ciphertext& ctxt,
    const int slots,
    const bool prescaled,
    const uint32_t numIterations,
    const uint32_t precision);
void BootstrapStCFirst(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void BootstrapStCFirst(Ciphertext& ctxt,
    const int slots,
    const bool prescaled,
    const uint32_t numIterations,
    const uint32_t precision);

/// Fourier-extension functional bootstrapping: bit-exact transcription of
/// FHECKKSRNS::EvalFEFuncBootstrap. Refreshes `ctxt` in place and evaluates the Fourier series
/// `coefficients` (c_0 first) on its message, which must lie in [-1/2, 1/2): the result is
/// 2 * Re(sum_j c_j exp(2 * pi * i * j * m / 2)). Requires an EvalFEFuncBootstrapSetup
/// precomputation for `slots`. Every argument is validated before `ctxt` is touched.
void FEFuncBootstrap(Ciphertext& ctxt, int slots, const std::vector<std::complex<double>>& coefficients);
/// FHECKKSRNS::EvalFEFuncBootstrapPrecompute: the function-independent part of FEFuncBootstrap.
/// Consumes `ctxt` (left holding exp(2 * pi * i * m / 2)) and returns the powers needed to
/// evaluate `coefficients` (degree >= 5) and any series of up to the same Paterson-Stockmeyer
/// capacity.
SeriesPowers FEFuncBootstrapPrecompute(Ciphertext& ctxt, int slots, const std::vector<std::complex<double>>& coefficients);
/// FHECKKSRNS::EvalFEFuncBootstrapWithPrecomp: evaluates one Fourier series on powers from
/// FEFuncBootstrapPrecompute. Never mutates `powers`.
Ciphertext FEFuncBootstrapWithPrecomp(SeriesPowers& powers, const std::vector<std::complex<double>>& coefficients);
/// Throws std::invalid_argument unless `coefficients` is a Fourier series the FE functional
/// bootstrap can evaluate: at least two entries, all finite, and a nonzero coefficient above the
/// constant term. Unless `trimmed` (FEFuncBootstrapWithPrecomp, which drops trailing zeros first),
/// a series below degree 5 must also have a nonzero last entry (the OpenFHE reference
/// dereferences a missing power on it, after the whole refresh). `who` prefixes the message. Exposed so that callers can reject a
/// series before choosing a backend.
void ValidateFEFuncCoefficients(const std::vector<std::complex<double>>& coefficients, const char* who, bool trimmed);

double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false, bool sparse_encaps = false);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAP_CUH