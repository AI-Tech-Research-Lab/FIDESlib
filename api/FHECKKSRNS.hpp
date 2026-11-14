#ifndef API_FHECKKSRNS_HPP
#define API_FHECKKSRNS_HPP

#include <complex>
#include <cstdint>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Static helpers of the CKKS bootstrapping scheme, mirroring lbcrypto::FHECKKSRNS.
struct FHECKKSRNS {
    /// @brief Multiplicative depth consumed by EvalFEFuncBootstrap (and by
    /// EvalFEFuncBootstrapPrecompute + EvalFEFuncBootstrapWithPrecomp for the same series):
    /// SlotsToCoeffs, CoeffsToSlots, the complex exponential and its double-angle iterations
    /// (selected by the secret key distribution and, for SPARSE_ENCAPSULATED, by firstModSize),
    /// and the Fourier series. Forwards to lbcrypto::FHECKKSRNS::GetFEFBTDepth.
    /// @param levelBudget Levels spent on CoeffsToSlots and SlotsToCoeffs (two entries).
    /// @param coefficients Fourier coefficients of the function (c_0 first); for a family of
    /// functions sharing one precomputation, pass the longest series.
    /// @param skd Secret key distribution of the context.
    /// @param firstModSize Size of the first modulus in bits (only used for SPARSE_ENCAPSULATED).
    static uint32_t GetFEFBTDepth(const std::vector<uint32_t>& levelBudget,
        const std::vector<std::complex<double>>& coefficients,
        SecretKeyDist skd = SPARSE_TERNARY,
        uint32_t firstModSize = 60);
};

} // namespace fideslib

#endif // API_FHECKKSRNS_HPP
