#include "FHECKKSRNS.hpp"

#include <openfhe.h>
#include <scheme/ckksrns/ckksrns-fhe.h>

namespace fideslib {

uint32_t FHECKKSRNS::GetFEFBTDepth(const std::vector<uint32_t>& levelBudget,
    const std::vector<std::complex<double>>& coefficients,
    SecretKeyDist skd,
    uint32_t firstModSize) {
    // OpenFHE indexes levelBudget[0] and [1] unchecked.
    if (levelBudget.size() != 2) {
        OPENFHE_THROW("GetFEFBTDepth: levelBudget must have exactly 2 entries (CoeffsToSlots, SlotsToCoeffs).");
    }
    if (coefficients.empty()) {
        OPENFHE_THROW("GetFEFBTDepth: the coefficients vector can not be empty.");
    }

    // Same mapping as CCParams::SetSecretKeyDist.
    lbcrypto::SecretKeyDist dist;
    switch (skd) {
    case SecretKeyDist::UNIFORM_TERNARY: dist = lbcrypto::UNIFORM_TERNARY; break;
    case SecretKeyDist::SPARSE_TERNARY: dist = lbcrypto::SPARSE_TERNARY; break;
    case SecretKeyDist::SPARSE_ENCAPSULATED: dist = lbcrypto::SPARSE_ENCAPSULATED; break;
    default: OPENFHE_THROW("GetFEFBTDepth: FE functional bootstrapping needs a UNIFORM_TERNARY, SPARSE_TERNARY or SPARSE_ENCAPSULATED secret key.");
    }

    return lbcrypto::FHECKKSRNS::GetFEFBTDepth(levelBudget, coefficients, dist, firstModSize);
}

} // namespace fideslib
