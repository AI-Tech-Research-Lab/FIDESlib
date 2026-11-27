#ifndef API_PLAINTEXT_HPP
#define API_PLAINTEXT_HPP

#include <any>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#include "Definitions.hpp"

namespace FIDESlib::CKKS {
struct RawPlainCoefficients;
}

namespace fideslib {
/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl {
  public:
    PlaintextImpl() = default;
    ~PlaintextImpl();

    PlaintextImpl(const CryptoContext<DCRTPoly>&& context);

    // ---- Copy ----

    PlaintextImpl(const PlaintextImpl&);
    PlaintextImpl(const Plaintext&);
    PlaintextImpl& operator=(const PlaintextImpl&) = delete;
    PlaintextImpl& operator=(const Plaintext&) = delete;

    // ---- Move ----

    PlaintextImpl(PlaintextImpl&&) = delete;
    PlaintextImpl(Plaintext&&) = delete;
    PlaintextImpl& operator=(PlaintextImpl&&) = delete;
    PlaintextImpl& operator=(Plaintext&&) = delete;

    // ---- Functions ----

    void SetLength(size_t length);
    void SetSlots(uint32_t slots);

    double GetLogPrecision() const;
    uint32_t GetLevel() const;
    uint32_t GetSlots() const;
    std::vector<std::complex<double>> GetCKKSPackedValue() const;
    std::vector<double> GetRealPackedValue() const;

    /// @brief Make sure `cpu` holds a fully encoded OpenFHE plaintext. A plaintext made with
    /// GPU encoding enabled (CryptoContextImpl::SetGPUPlaintextEncoding) keeps its OpenFHE
    /// plaintext unencoded until something on the CPU needs its element; this encodes it then
    /// (once, thread-safe). A no-op for every other plaintext.
    void EnsureCpuEncoding() const;

    // ---- Friend Operators ----

    friend std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt);

    // ---- Internal State ----

    std::any cpu;
    uint32_t gpu = 0;
    /// @brief Flag indicating whether the plaintext is loaded to the devices.
    bool loaded = false;
    /// @brief Parent context for device management.
    CryptoContext<DCRTPoly> parent_context;
    /// @brief Rounded encoding coefficients the GPU finishes encoding from (see
    /// CryptoContextImpl::SetGPUPlaintextEncoding); null for plaintexts encoded on the CPU.
    std::shared_ptr<const FIDESlib::CKKS::RawPlainCoefficients> coefficients;
};

// ---- Override Operators ----

std::ostream& operator<<(std::ostream& os, const Plaintext& pt);
bool operator==(const PlaintextImpl& lhs, const PlaintextImpl& rhs);
bool operator!=(const PlaintextImpl& lhs, const PlaintextImpl& rhs);
} // namespace fideslib

#endif // API_PLAINTEXT_HPP