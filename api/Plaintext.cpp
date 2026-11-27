#include "Plaintext.hpp"
#include "CryptoContext.hpp"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <openfhe.h>

#include <mutex>

namespace fideslib {
namespace {
std::mutex cpu_encoding_mutex;
}

bool PlaintextImpl::HasGPUEncoding() const {
    if (!this->coefficients)
        return false;
    const auto* impl = std::any_cast<lbcrypto::Plaintext>(&this->cpu);
    const auto source = this->coefficients_source.lock();
    return impl != nullptr && *impl && source && source.get() == static_cast<const void*>(impl->get());
}

void PlaintextImpl::EnsureCpuEncoding() const {
    // Only the OpenFHE plaintext the coefficients came from is ours to encode; one that replaced
    // it in `cpu` is left as it is.
    if (!this->HasGPUEncoding())
        return;
    const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
    std::lock_guard<std::mutex> lock(cpu_encoding_mutex);
    if (impl && !impl->IsEncoded()) {
        impl->Encode();
        // NoiseLevel is the degree of the encoded plaintext, which OpenFHE can set after Encode()
        // (FLEXIBLEAUTOEXT at level 0).
        if (impl->GetNoiseScaleDeg() != static_cast<size_t>(this->coefficients->NoiseLevel))
            impl->SetNoiseScaleDeg(this->coefficients->NoiseLevel);
    }
}

PlaintextImpl::PlaintextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
    if (!context) {
        OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
    }
}

PlaintextImpl::~PlaintextImpl() {
    if (this->loaded && this->gpu != 0 && this->parent_context) {
        this->parent_context->EvictDevicePlaintext(this->gpu);
        this->gpu = 0;
    }
}

// ---- Functions ----

void PlaintextImpl::SetLength(size_t length) {
    this->EnsureCpuEncoding();
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->cpu);
        impl->SetLength(length);
    }
}

void PlaintextImpl::SetSlots(uint32_t slots) {
    // Encode first: on an unencoded OpenFHE plaintext the slot count would change the encoding.
    this->EnsureCpuEncoding();
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->cpu);
        impl->SetSlots(slots);
    }
}

double PlaintextImpl::GetLogPrecision() const {
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
        return impl->GetLogPrecision();
    }
    return 0.0;
}

uint32_t PlaintextImpl::GetLevel() const {
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
        return impl->GetLevel();
    }
    return 0;
}

uint32_t PlaintextImpl::GetSlots() const {
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
        return impl->GetSlots();
    }
    return 0;
}

std::vector<std::complex<double>> PlaintextImpl::GetCKKSPackedValue() const {
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
        return impl->GetCKKSPackedValue();
    }
    return {};
}

std::vector<double> PlaintextImpl::GetRealPackedValue() const {
    if (this->cpu.has_value()) {
        auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
        return impl->GetRealPackedValue();
    }
    return {};
}

// ---- Friend Operators ----

std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt) {
    if (pt.cpu.has_value()) {
        const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt.cpu);
        os << impl;
    } else {
        os << "Empty Plaintext";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Plaintext& pt) {
    if (pt && pt->cpu.has_value()) {
        const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
        os << impl;
    } else {
        os << "Empty Plaintext";
    }
    return os;
}

bool operator==(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
    lhs.EnsureCpuEncoding();
    rhs.EnsureCpuEncoding();
    if (!lhs.cpu.has_value() || !rhs.cpu.has_value())
        return lhs.cpu.has_value() == rhs.cpu.has_value();
    const auto& l = std::any_cast<const lbcrypto::Plaintext&>(lhs.cpu);
    const auto& r = std::any_cast<const lbcrypto::Plaintext&>(rhs.cpu);
    if (!l || !r)
        return l == r;
    return *l == *r;
}

bool operator!=(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
    return !(lhs == rhs);
}
} // namespace fideslib