#ifndef API_SERIESPOWERS_HPP
#define API_SERIESPOWERS_HPP

#include <any>
#include <memory>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Powers of the complex exponential produced by
/// CryptoContextImpl::EvalFEFuncBootstrapPrecompute; the fideslib counterpart of
/// lbcrypto::seriesPowers. Pass it to EvalFEFuncBootstrapWithPrecomp to evaluate one function each.
///
/// Built on the GPU (after LoadContext()) it holds the powers in GPU memory until destroyed. They
/// do not go through the ciphertext registry: they are not counted in the SetCiphertextCache()
/// budget and cannot be offloaded to host RAM. The handle keeps its context alive, so the GPU
/// memory is always released before the context is torn down.
struct SeriesPowersImpl {
    /// @brief API ciphertext holding exp(2*pi*i*t): the template every result is cloned from, and
    /// the reference that keeps the parent context alive. Declared first so it is destroyed last.
    Ciphertext<DCRTPoly> exponential;
    /// @brief std::shared_ptr<lbcrypto::seriesPowers<lbcrypto::DCRTPoly>> when built on the CPU.
    std::any cpu;
    /// @brief FIDESlib::CKKS::SeriesPowers when built on the GPU.
    std::shared_ptr<void> gpu;
    /// @brief Whether the powers were built on the GPU.
    bool loaded = false;

    ~SeriesPowersImpl() {
        // Free the powers before `exponential` can release the last reference to the context.
        gpu.reset();
        cpu.reset();
    }
};

/// @brief Shared pointer alias for SeriesPowersImpl.
using SeriesPowers = std::shared_ptr<SeriesPowersImpl>;

} // namespace fideslib

#endif // API_SERIESPOWERS_HPP
