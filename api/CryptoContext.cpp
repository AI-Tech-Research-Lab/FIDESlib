#include "CryptoContext.hpp"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CudaUtils.cuh"
#include "Definitions.hpp"
#include "PolyApprox.cuh"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "ciphertext-fwd.h"
#include "cryptocontext-fwd.h"
#include "lattice/hal/lat-backend.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <openfhe.h>
// Serialization headers - required for cereal type registration.
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

template <> std::map<std::string, std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
template <>
std::map<std::string, std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalAutomorphismKeyMap;

namespace fideslib {

namespace {
inline void EnsureMutableCpuCiphertext(Ciphertext<DCRTPoly>& ct) {
    if (ct->need_lazy_copy) {
        ct->EnsureLazyCPUCopy();
    }
}
} // namespace

static std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
    { .p = 2251799661248513 },
    { .p = 2251799661641729 },
    { .p = 2251799665180673 },
    { .p = 2251799682088961 },
    { .p = 2251799678943233 },
    { .p = 2251799717609473 },
    { .p = 2251799710138369 },
    { .p = 2251799708827649 },
    { .p = 2251799707385857 },
    { .p = 2251799713677313 },
    { .p = 2251799712366593 },
    { .p = 2251799716691969 },
    { .p = 2251799714856961 },
    { .p = 2251799726522369 },
    { .p = 2251799726129153 },
    { .p = 2251799747493889 },
    { .p = 2251799741857793 },
    { .p = 2251799740416001 },
    { .p = 2251799746707457 },
    { .p = 2251799756013569 },
    { .p = 2251799775805441 },
    { .p = 2251799763091457 },
    { .p = 2251799767154689 },
    { .p = 2251799765975041 },
    { .p = 2251799770562561 },
    { .p = 2251799769776129 },
    { .p = 2251799772266497 },
    { .p = 2251799775281153 },
    { .p = 2251799774887937 },
    { .p = 2251799797432321 },
    { .p = 2251799787995137 },
    { .p = 2251799787601921 },
    { .p = 2251799791403009 },
    { .p = 2251799789568001 },
    { .p = 2251799795466241 },
    { .p = 2251799807131649 },
    { .p = 2251799806345217 },
    { .p = 2251799805165569 },
    { .p = 2251799813554177 },
    { .p = 2251799809884161 },
    { .p = 2251799810670593 },
    { .p = 2251799818928129 },
    { .p = 2251799816568833 },
    { .p = 2251799815520257 } };

static std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
    { .p = 2305843009220116481 },
    { .p = 2305843009221820417 },
    { .p = 2305843009224179713 },
    { .p = 2305843009225228289 },
    { .p = 2305843009227980801 },
    { .p = 2305843009229160449 },
    { .p = 2305843009229946881 },
    { .p = 2305843009231650817 },
    { .p = 2305843009235189761 },
    { .p = 2305843009240301569 },
    { .p = 2305843009242923009 },
    { .p = 2305843009244889089 },
    { .p = 2305843009245413377 },
    { .p = 2305843009247641601 } };

static std::unordered_map<PKESchemeFeature, lbcrypto::PKESchemeFeature> PKESchemeFeatureMap = {
    { PKESchemeFeature::PKE, lbcrypto::PKE },
    { PKESchemeFeature::KEYSWITCH, lbcrypto::KEYSWITCH },
    { PKESchemeFeature::PRE, lbcrypto::PRE },
    { PKESchemeFeature::LEVELEDSHE, lbcrypto::LEVELEDSHE },
    { PKESchemeFeature::ADVANCEDSHE, lbcrypto::ADVANCEDSHE },
    { PKESchemeFeature::MULTIPARTY, lbcrypto::MULTIPARTY },
    { PKESchemeFeature::FHE, lbcrypto::FHE },
    { PKESchemeFeature::SCHEMESWITCH, lbcrypto::SCHEMESWITCH },
};

CryptoContextImpl<DCRTPoly>::~CryptoContextImpl() {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->loaded) {
        auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
        FIDESlib::CKKS::DeregisterCryptoContextGPU(context_gpu);
        this->gpu = std::any();
    }
    // Deliberately no ClearEvalMultKeys()/ClearEvalAutomorphismKeys(): the no-argument overloads
    // wipe OpenFHE's *global* static key maps, i.e. every other live context's keys too. OpenFHE's
    // own CryptoContextImpl has no such destructor; free host key memory with the per-context
    // overloads ClearEvalMultKeys(cc) / ClearEvalAutomorphismKeys(cc) when actually needed.
}

// ---- Enable features ----

void CryptoContextImpl<DCRTPoly>::Enable(PKESchemeFeature feature) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    context->Enable(PKESchemeFeatureMap[feature]);
}

void CryptoContextImpl<DCRTPoly>::Enable(uint32_t featureMask) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    context->Enable(featureMask);
}

// ---- Getters ----

uint32_t CryptoContextImpl<DCRTPoly>::GetCyclotomicOrder() const {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    return context->GetCyclotomicOrder();
}

uint32_t CryptoContextImpl<DCRTPoly>::GetRingDimension() const {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    return context->GetRingDimension();
}

double CryptoContextImpl<DCRTPoly>::GetPreScaleFactor(uint32_t slots) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (!this->loaded) {
        FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }
    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    return FIDESlib::CKKS::GetPreScaleFactor(context_gpu, static_cast<int32_t>(slots));
}

// ---- Setters ----

void CryptoContextImpl<DCRTPoly>::SetAutoLoadPlaintexts(bool autoload) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    this->auto_load_plaintexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetAutoLoadCiphertexts(bool autoload) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    this->auto_load_ciphertexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetDevices(const std::vector<int>& devices) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->loaded) {
        FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
        OPENFHE_THROW("SetDevices must be called before LoadContext");
    }

    this->devices = devices;
}

// ---- Rotation-key VRAM cache ----

void CryptoContextImpl<DCRTPoly>::SetRotationKeyCache(const size_t bytes) {
    FIDESlib::CudaNvtxRange r("API");
    this->rotation_key_cache_bytes = bytes;
    if (this->loaded) {
        auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
        context_gpu->SetRotationKeyCache(bytes);
    }
}

size_t CryptoContextImpl<DCRTPoly>::GetRotationKeyCache() const {
    return this->rotation_key_cache_bytes;
}

void CryptoContextImpl<DCRTPoly>::OffloadRotationKeys(const std::vector<int>& indexes) {
    FIDESlib::CudaNvtxRange r("API");
    if (!this->loaded) {
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }
    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    context_gpu->OffloadRotationKeys(indexes);
}

void CryptoContextImpl<DCRTPoly>::PinRotationKey(const int index, const bool pin) {
    FIDESlib::CudaNvtxRange r("API");
    if (!this->loaded) {
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }
    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    context_gpu->PinRotationKey(index, pin);
}

bool CryptoContextImpl<DCRTPoly>::IsRotationKeyResident(const int index) const {
    if (!this->loaded)
        return false;
    auto& context_gpu = std::any_cast<const FIDESlib::CKKS::Context&>(this->gpu);
    return context_gpu->IsRotationKeyResident(index);
}

size_t CryptoContextImpl<DCRTPoly>::GetRotationKeyCacheResidentBytes() const {
    if (!this->loaded)
        return 0;
    auto& context_gpu = std::any_cast<const FIDESlib::CKKS::Context&>(this->gpu);
    return context_gpu->RotationKeyCacheResidentBytes();
}

// ---- Plaintext VRAM cache ----

void CryptoContextImpl<DCRTPoly>::SetPlaintextCache(const size_t bytes) {
    FIDESlib::CudaNvtxRange r("API");
    this->plaintext_cache_bytes = bytes;
    this->EnforcePlaintextBudget();
}

size_t CryptoContextImpl<DCRTPoly>::GetPlaintextCache() const {
    return this->plaintext_cache_bytes;
}

size_t CryptoContextImpl<DCRTPoly>::GetPlaintextCacheResidentBytes() const {
    return this->plaintext_resident_bytes;
}

void CryptoContextImpl<DCRTPoly>::OffloadPlaintexts() {
    FIDESlib::CudaNvtxRange r("API");
    // Collect first: offloading mutates plaintext_cache.
    std::vector<uint32_t> handles;
    handles.reserve(this->plaintext_cache.size());
    for (const auto& [handle, entry] : this->plaintext_cache) {
        if (!this->plaintext_pinned.contains(handle)) {
            handles.push_back(handle);
        }
    }
    for (const uint32_t handle : handles) {
        this->OffloadPlaintextHandle(handle);
    }
}

void CryptoContextImpl<DCRTPoly>::PinPlaintext(Plaintext& pt, const bool pin) {
    FIDESlib::CudaNvtxRange r("API");
    PlaintextCacheHold plaintext_hold(*this);
    if (pin) {
        this->LoadPlaintext(pt);
        if (pt->loaded) {
            this->plaintext_pinned.insert(pt->gpu);
        }
    } else if (pt->gpu != 0) {
        this->plaintext_pinned.erase(pt->gpu);
        this->EnforcePlaintextBudget();
    }
}

void CryptoContextImpl<DCRTPoly>::TrackPlaintext(const uint32_t handle, const Plaintext& pt, const size_t bytes) {
    this->plaintext_lru.push_front(handle);
    this->plaintext_cache.emplace(handle, PlaintextCacheEntry{ std::weak_ptr<PlaintextImpl>(pt), bytes, this->plaintext_lru.begin() });
    this->plaintext_resident_bytes += bytes;
}

void CryptoContextImpl<DCRTPoly>::UntrackPlaintext(const uint32_t handle) {
    auto it = this->plaintext_cache.find(handle);
    if (it == this->plaintext_cache.end()) {
        return;
    }
    this->plaintext_resident_bytes -= std::min(it->second.bytes, this->plaintext_resident_bytes);
    this->plaintext_lru.erase(it->second.lru);
    this->plaintext_pinned.erase(handle);
    this->plaintext_cache.erase(it);
}

void CryptoContextImpl<DCRTPoly>::TouchPlaintext(const uint32_t handle) {
    auto it = this->plaintext_cache.find(handle);
    if (it == this->plaintext_cache.end()) {
        return;
    }
    this->plaintext_lru.splice(this->plaintext_lru.begin(), this->plaintext_lru, it->second.lru);
}

bool CryptoContextImpl<DCRTPoly>::OffloadPlaintextHandle(const uint32_t handle) {
    auto it = this->plaintext_cache.find(handle);
    if (it == this->plaintext_cache.end()) {
        return false;
    }
    if (const auto owner = it->second.owner.lock()) {
        // The next operation taking this plaintext re-uploads it from the host encoding.
        owner->loaded = false;
        owner->gpu = 0;
    }
    // Drops the registry's reference to the device plaintext (and its bookkeeping, through
    // UntrackPlaintext). If an operation in flight still holds a reference, the VRAM comes
    // back when that one dies instead of here.
    return this->EvictDevicePlaintext(handle);
}

void CryptoContextImpl<DCRTPoly>::EnforcePlaintextBudget(const uint32_t protect) {
    if (this->plaintext_cache_bytes == SIZE_MAX || this->plaintext_cache_hold > 0) {
        return;
    }
    while (this->plaintext_resident_bytes > this->plaintext_cache_bytes) {
        uint32_t victim = 0;
        for (auto it = this->plaintext_lru.rbegin(); it != this->plaintext_lru.rend(); ++it) {
            if (*it != protect && !this->plaintext_pinned.contains(*it)) {
                victim = *it;
                break;
            }
        }
        // Nothing left to evict: the budget is unsatisfiable (everything is pinned or in use),
        // so overshoot it rather than break the caller.
        if (victim == 0 || !this->OffloadPlaintextHandle(victim)) {
            return;
        }
    }
}

// ---- Ciphertext VRAM cache ----

void CryptoContextImpl<DCRTPoly>::SetCiphertextCache(const size_t bytes) {
    FIDESlib::CudaNvtxRange r("API");
    this->ciphertext_cache_bytes = bytes;
    this->EnforceCiphertextBudget();
}

size_t CryptoContextImpl<DCRTPoly>::GetCiphertextCache() const {
    return this->ciphertext_cache_bytes;
}

size_t CryptoContextImpl<DCRTPoly>::GetCiphertextCacheResidentBytes() const {
    return this->ciphertext_resident_bytes;
}

void CryptoContextImpl<DCRTPoly>::OffloadCiphertexts() {
    FIDESlib::CudaNvtxRange r("API");
    // Snapshot first: offloading mutates ciphertext_lru.
    const std::vector<uint32_t> handles(this->ciphertext_lru.begin(), this->ciphertext_lru.end());
    for (const uint32_t handle : handles) {
        if (!this->ciphertext_pinned.contains(handle)) {
            this->OffloadCiphertextForCache(handle);
        }
    }
}

void CryptoContextImpl<DCRTPoly>::PinCiphertext(Ciphertext<DCRTPoly>& ct, const bool pin) {
    FIDESlib::CudaNvtxRange r("API");
    if (pin) {
        this->LoadCiphertext(ct);
        if (ct->loaded && ct->gpu != 0) {
            // Pinned means resident, so bring it back if it was parked in host RAM.
            this->ReloadCiphertext(ct->gpu);
            this->ciphertext_pinned.insert(ct->gpu);
        }
    } else if (ct->gpu != 0) {
        this->ciphertext_pinned.erase(ct->gpu);
        this->EnforceCiphertextBudget();
    }
}

void CryptoContextImpl<DCRTPoly>::TrackCiphertext(const uint32_t handle, const size_t bytes) {
    this->ciphertext_lru.push_front(handle);
    this->ciphertext_cache.emplace(handle, CiphertextCacheEntry{ bytes, true, this->ciphertext_lru.begin() });
    this->ciphertext_resident_bytes += bytes;
}

void CryptoContextImpl<DCRTPoly>::UntrackCiphertext(const uint32_t handle) {
    auto it = this->ciphertext_cache.find(handle);
    if (it == this->ciphertext_cache.end()) {
        return;
    }
    if (it->second.resident) {
        this->ciphertext_resident_bytes -= std::min(it->second.bytes, this->ciphertext_resident_bytes);
        this->ciphertext_lru.erase(it->second.lru);
    }
    this->ciphertext_pinned.erase(handle);
    this->ciphertext_cache.erase(it);
}

void CryptoContextImpl<DCRTPoly>::TouchCiphertext(const uint32_t handle, const size_t bytes) {
    auto it = this->ciphertext_cache.find(handle);
    if (it == this->ciphertext_cache.end()) {
        return;
    }
    CiphertextCacheEntry& entry = it->second;
    if (entry.resident) {
        this->ciphertext_resident_bytes -= std::min(entry.bytes, this->ciphertext_resident_bytes);
        this->ciphertext_lru.splice(this->ciphertext_lru.begin(), this->ciphertext_lru, entry.lru);
    } else {
        this->ciphertext_lru.push_front(handle);
        entry.lru = this->ciphertext_lru.begin();
        entry.resident = true;
    }
    this->ciphertext_resident_bytes += bytes;
    entry.bytes = bytes;
}

void CryptoContextImpl<DCRTPoly>::MarkCiphertextOffloaded(const uint32_t handle) {
    auto it = this->ciphertext_cache.find(handle);
    if (it == this->ciphertext_cache.end() || !it->second.resident) {
        return;
    }
    this->ciphertext_resident_bytes -= std::min(it->second.bytes, this->ciphertext_resident_bytes);
    this->ciphertext_lru.erase(it->second.lru);
    it->second.resident = false;
}

bool CryptoContextImpl<DCRTPoly>::OffloadCiphertextForCache(const uint32_t handle) {
    auto it = this->ciphertext_cache.find(handle);
    if (it == this->ciphertext_cache.end() || !it->second.resident) {
        return false;
    }
    auto reg = this->device_ciphertexts.find(handle);
    if (reg == this->device_ciphertexts.end()) {
        return false;
    }
    // Bypass GetDeviceCiphertext()'s auto-reload: it exists precisely to undo this state.
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(reg->second);
    if (ct_gpu->isOffloaded()) {
        // Offloaded behind the cache's back (nothing to free, just stale accounting).
        this->MarkCiphertextOffloaded(handle);
        return true;
    }
    // A ciphertext raised to the extended basis mid-key-switch cannot be snapshotted --
    // Ciphertext::offload() asserts against it -- so skip it and let the caller try another.
    if (ct_gpu->c0.isModUp() || ct_gpu->c1.isModUp()) {
        return false;
    }
    ct_gpu->offload();
    this->MarkCiphertextOffloaded(handle);
    return true;
}

void CryptoContextImpl<DCRTPoly>::EnforceCiphertextBudget(const uint32_t protect) {
    if (this->ciphertext_cache_bytes == SIZE_MAX || this->ciphertext_resident_bytes <= this->ciphertext_cache_bytes) {
        return;
    }
    // One pass, least recently used first, over a snapshot (offloading mutates the list). If
    // it ends still over budget -- everything left is pinned, in the extended basis, or the
    // budget is smaller than the ciphertext in use -- overshoot rather than break the caller.
    const std::vector<uint32_t> candidates(this->ciphertext_lru.rbegin(), this->ciphertext_lru.rend());
    for (const uint32_t handle : candidates) {
        if (this->ciphertext_resident_bytes <= this->ciphertext_cache_bytes) {
            break;
        }
        if (handle == protect || this->ciphertext_pinned.contains(handle)) {
            continue;
        }
        this->OffloadCiphertextForCache(handle);
    }
}

// ---- Bootstrap-precomputation VRAM cache ----

void CryptoContextImpl<DCRTPoly>::SetBootstrapCache(const size_t bytes) {
    FIDESlib::CudaNvtxRange r("API");
    this->bootstrap_cache_bytes = bytes;
    if (this->loaded) {
        auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
        context_gpu->SetBootstrapCache(bytes);
    }
}

size_t CryptoContextImpl<DCRTPoly>::GetBootstrapCache() const {
    return this->bootstrap_cache_bytes;
}

size_t CryptoContextImpl<DCRTPoly>::GetBootstrapCacheResidentBytes() const {
    if (!this->loaded)
        return 0;
    auto& context_gpu = std::any_cast<const FIDESlib::CKKS::Context&>(this->gpu);
    return context_gpu->BootstrapCacheResidentBytes();
}

size_t CryptoContextImpl<DCRTPoly>::GetBootstrapCacheLoadedBytes() const {
    if (!this->loaded)
        return 0;
    auto& context_gpu = std::any_cast<const FIDESlib::CKKS::Context&>(this->gpu);
    return context_gpu->BootstrapCacheLoadedBytes();
}

void CryptoContextImpl<DCRTPoly>::OffloadBootstrapPrecomputation() {
    FIDESlib::CudaNvtxRange r("API");
    if (!this->loaded) {
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }
    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    context_gpu->OffloadBootstrapPrecomputation();
}

// ---- Load to devices ----

void CryptoContextImpl<DCRTPoly>::LoadContext(const PublicKey<DCRTPoly>& publicKey) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->loaded || this->devices.empty())
        return;

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    FIDESlib::CKKS::Parameters params{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

    // Determine the boot configuration based on the secret key distribution.
    const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
    FIDESlib::BOOT_CONFIG bootConfig;
    switch (this->keyDist) {
    case fideslib::UNIFORM_TERNARY: bootConfig = FIDESlib::UNIFORM; break;
    case fideslib::SPARSE_TERNARY: bootConfig = FIDESlib::SPARSE; break;
    case fideslib::SPARSE_ENCAPSULATED: bootConfig = FIDESlib::ENCAPS; break;
    default: bootConfig = FIDESlib::UNIFORM; break;
    }

    FIDESlib::CKKS::RawParams rawParams = FIDESlib::CKKS::GetRawParams(context, bootConfig);
    params = params.adaptTo(rawParams);

    // The Context has to be moved into this->gpu *before* the keys are built, and the keys must
    // then be built against that stored instance: KeySwitchingKey (and Ciphertext) keep a
    // `Context&`, so binding them to a local that is later moved out of leaves every stored key
    // holding a dangling reference. A permanently resident key never dereferences it again, which
    // is why building against a local used to appear to work -- but a rotation key under a VRAM
    // budget reloads itself on first use and goes through that reference to do it.
    this->gpu = std::make_any<FIDESlib::CKKS::Context>(FIDESlib::CKKS::GenCryptoContextGPU(params, this->devices));
    FIDESlib::CKKS::Context& c = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);

    // Must happen before the keys below are built: a finite budget makes every rotation key
    // (including the bootstrapping ones) start as a host-RAM snapshot that loads lazily and can
    // be evicted again. Keys built without it are permanently resident.
    c->SetRotationKeyCache(this->rotation_key_cache_bytes);
    // Same for the bootstrap precomputation: under a finite budget its matrices are built with a
    // host snapshot and offloaded right away (AddBootstrapPrecomputation below).
    c->SetBootstrapCache(this->bootstrap_cache_bytes);

    auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);

    // Multiplicative key switching key.
    auto& keyMap = context->GetAllEvalMultKeys(); // lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(pkImpl->GetKeyTag()) != keyMap.end()) {
        auto raw_eval_ksk = FIDESlib::CKKS::GetEvalKeySwitchKey(pkImpl);
        FIDESlib::CKKS::KeySwitchingKey eval_ksk(c);
        eval_ksk.Initialize(raw_eval_ksk);
        c->AddEvalKey(std::move(eval_ksk));
    }
    // Rotational key switching keys.
    for (const auto& step : this->rotation_indexes) {
        auto raw_rot_ksk = FIDESlib::CKKS::GetRotationKeySwitchKey(pkImpl, step);
        FIDESlib::CKKS::KeySwitchingKey rot_ksk(c);
        rot_ksk.Initialize(raw_rot_ksk, c->rotationKeysLazy());
        c->AddRotationKey(step, std::move(rot_ksk));
    }
    // Bootstrapping keys.
    for (const auto& slot : this->slots_bootstrap) {
        FIDESlib::CKKS::AddBootstrapPrecomputation(pkImpl, slot, c);
    }

    this->loaded = true;
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (pt->loaded || this->devices.empty())
        return;

    if (!this->loaded) {
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }

    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
    FIDESlib::CKKS::RawPlainText raw_pt = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
    std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu, raw_pt);
    const size_t bytes = gpu_pt->limbBytes();
    uint32_t handle = this->RegisterDevicePlaintext(std::move(gpu_pt));
    pt->gpu = handle;
    pt->loaded = true;

    // A plaintext's size is only known once it is built, so the budget is enforced after the
    // upload (protecting the new plaintext, which the caller is about to use) rather than
    // before it: it can overshoot by this one plaintext until the next load.
    this->TrackPlaintext(handle, pt, bytes);
    this->EnforcePlaintextBudget(handle);
}

void CryptoContextImpl<DCRTPoly>::LoadCiphertext(Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->devices.empty())
        return;

    // This is the ciphertext cache's only automatic eviction point, and the reason it is safe:
    // every operation calls LoadCiphertext for its operands before it takes a single device
    // pointer, so whatever gets evicted here is something the operation has not fetched yet
    // (and would transparently reload). Already-loaded operands come through here too -- the
    // early return below -- which is what keeps the budget binding in a loop of operations
    // over ciphertexts that are all resident already.
    if (ct->loaded) {
        if (ct->gpu != 0) {
            this->ReloadCiphertext(ct->gpu); // no-op unless it was parked in host RAM
            this->EnforceCiphertextBudget(ct->gpu);
        }
        return;
    }

    if (!this->loaded) {
        OPENFHE_THROW("CryptoContext not loaded to any device");
    }

    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    const auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
    FIDESlib::CKKS::RawCipherText raw_ct = FIDESlib::CKKS::GetRawCipherText(context, ctImpl);
    std::shared_ptr<FIDESlib::CKKS::Ciphertext> gpu_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
    uint32_t handle = this->RegisterDeviceCiphertext(std::move(gpu_ct));
    ct->gpu = handle;
    ct->loaded = true;
    ct->original_level = this->multiplicative_depth - ct->GetLevel();

    this->EnforceCiphertextBudget(handle);
}

void CryptoContextImpl<DCRTPoly>::OffloadCiphertext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API");
    if (handle == 0) {
        return;
    }
    auto it = device_ciphertexts.find(handle);
    if (it == device_ciphertexts.end()) {
        return;
    }
    // Bypass GetDeviceCiphertext()'s auto-reload: it exists precisely to undo this state.
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(it->second);
    ct_gpu->offload();
    this->MarkCiphertextOffloaded(handle);
}

void CryptoContextImpl<DCRTPoly>::ReloadCiphertext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API");
    if (handle == 0) {
        return;
    }
    auto it = device_ciphertexts.find(handle);
    if (it == device_ciphertexts.end()) {
        return;
    }
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(it->second);
    if (!ct_gpu->isOffloaded()) {
        return;
    }
    ct_gpu->reload();
    this->TouchCiphertext(handle, ct_gpu->limbBytes());
}

void CryptoContextImpl<DCRTPoly>::TrimGPUMemoryPool() {
    FIDESlib::CudaNvtxRange r("API");
    if (!this->loaded) {
        return;
    }
    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    FIDESlib::CKKS::TrimGPUMemoryPool(context_gpu);
}

bool CryptoContextImpl<DCRTPoly>::IsCiphertextOffloaded(uint32_t handle) const {
    FIDESlib::CudaNvtxRange r("API");
    if (handle == 0) {
        return false;
    }
    auto it = device_ciphertexts.find(handle);
    if (it == device_ciphertexts.end()) {
        return false;
    }
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(it->second);
    return ct_gpu->isOffloaded();
}

void CryptoContextImpl<DCRTPoly>::UnloadCiphertext(Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (!ct->loaded)
        return;

    // Materialize the host copy while the GPU still holds the data. The GPU is the
    // source of truth while loaded; EnsureUpToDateCPUCopy() also grows the CPU basis
    // for extended/modUp results (e.g. after EvalFastRotationExt) via the R15 path.
    ct->EnsureUpToDateCPUCopy();

    if (!this->EvictDeviceCiphertext(ct->gpu)) {
        OPENFHE_THROW("Ciphertext eviction error: could not evict Ciphertext from device");
    }
    ct->loaded = false;
    ct->gpu = 0;
}

// ---- Key Generation ----

KeyPair<DCRTPoly> CryptoContextImpl<DCRTPoly>::KeyGen() {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto keys = context->KeyGen();

    KeyPair<DCRTPoly> keypair;
    keypair.publicKey = std::make_shared<PublicKeyImpl<DCRTPoly>>();
    keypair.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();

    keypair.publicKey->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(keys.publicKey);
    keypair.secretKey->pimpl = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(keys.secretKey);

    return keypair;
}

void CryptoContextImpl<DCRTPoly>::EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (!this->devices.empty() && this->loaded) {
        OPENFHE_THROW("EvalMultKeyGen must be called before LoadContext");
    }

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
    context->EvalMultKeyGen(skImpl);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (!this->devices.empty() && this->loaded) {
        OPENFHE_THROW("EvalRotateKeyGen must be called before LoadContext");
    }

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
    context->EvalRotateKeyGen(skImpl, steps);
    this->rotation_indexes.insert(this->rotation_indexes.end(), steps.begin(), steps.end());
}

// ---- Bootstrapping ----

uint32_t CryptoContextImpl<DCRTPoly>::ResolveBootstrapSlots(uint32_t slots) const {
    if (slots != 0) {
        return slots;
    }
    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    return context->GetCyclotomicOrder() / 4;
}

void CryptoContextImpl<DCRTPoly>::RecordBootstrapMode(uint32_t slot, fideslib::BootstrapMode mode) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    CheckBootstrapModeConflict(slot, mode);
    bootstrap_mode[slot] = mode;
}

void CryptoContextImpl<DCRTPoly>::CheckBootstrapModeConflict(uint32_t slot, fideslib::BootstrapMode mode) const {
    auto it = bootstrap_mode.find(slot);
    if (it != bootstrap_mode.end() && it->second != fideslib::BootstrapMode::UNCONFIGURED && it->second != mode) {
        OPENFHE_THROW("Bootstrap setup conflict for slots=" + std::to_string(slot) + ": already configured for " +
            std::string(fideslib::BootstrapModeName(it->second)) + "; cannot reconfigure it for " + std::string(fideslib::BootstrapModeName(mode)) +
            ". The standard and SlotsToCoeffs-first bootstraps share one precomputation "
            "entry per slot count.");
    }
}

void CryptoContextImpl<DCRTPoly>::CheckBootstrapMode(uint32_t slot, std::initializer_list<fideslib::BootstrapMode> allowed, const char* who) const {
    auto it = bootstrap_mode.find(slot);
    if (it == bootstrap_mode.end() || it->second == fideslib::BootstrapMode::UNCONFIGURED) {
        OPENFHE_THROW(std::string(who) + ": no bootstrapping precomputation for slots=" + std::to_string(slot) + " (run the matching setup before LoadContext).");
    }
    for (auto m : allowed) {
        if (it->second == m) {
            return;
        }
    }
    OPENFHE_THROW(std::string(who) + ": slots=" + std::to_string(slot) + " is configured for " + std::string(fideslib::BootstrapModeName(it->second)) +
        ", which is incompatible with this evaluation.");
}

void CryptoContextImpl<DCRTPoly>::RequireBootstrapPrecomputation(uint32_t slot, const char* who) const {
    auto it = bootstrap_mode.find(slot);
    if (it == bootstrap_mode.end() || it->second == fideslib::BootstrapMode::UNCONFIGURED) {
        OPENFHE_THROW(std::string(who) + ": no bootstrapping precomputation for slots=" + std::to_string(slot) + " (run the matching setup before LoadContext).");
    }
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor, bool precompute, bool btsfirstboot) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Only before loading one must compute the bootstrapping auxiliary data.
    if (this->loaded) {
        OPENFHE_THROW("Context is already loaded");
    }
    // Guardrail: reject a conflicting mode before the reference setup overwrites the shared entry.
    CheckBootstrapModeConflict(ResolveBootstrapSlots(slots), btsfirstboot ? fideslib::BootstrapMode::STC_FIRST : fideslib::BootstrapMode::STANDARD);
    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

    std::vector<double> coeffchebyshev;
    int doubleAngleIts = 3;

    if (this->keyDist == fideslib::SPARSE_ENCAPSULATED) {
        coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated;
        doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
    } else if (this->keyDist == fideslib::SPARSE_TERNARY) {
        coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
        doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
    } else if (this->keyDist == fideslib::UNIFORM_TERNARY) {
        coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
        doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
    } else {
        OPENFHE_THROW("Unsupported key distribution");
    }

    int32_t modall = static_cast<int>(lbcrypto::GetMultiplicativeDepthByCoeffVector(coeffchebyshev, false)) + doubleAngleIts;

    if (this->devices.empty()) {
        // The bootstrap precomputation lives in the (CPU) OpenFHE context and is read back by
        // AddBootstrapPrecomputation/LoadContext, so BTSlotsEncoding must be forwarded here too
        // (the GPU-branch call below already passes it). modevallevels keeps the historical
        // default (std::numeric_limits<uint32_t>::max()) so existing non-BTS setups are untouched.
        context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, true, btsfirstboot, std::numeric_limits<uint32_t>::max());
    } else {
        context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, precompute, btsfirstboot, modall);
    }

    // Guardrail: one precomputation mode per slot count (the standard and StC-first setups share
    // the OpenFHE entry; a conflicting earlier setup is rejected by CheckBootstrapModeConflict).
    RecordBootstrapMode(ResolveBootstrapSlots(slots), btsfirstboot ? fideslib::BootstrapMode::STC_FIRST : fideslib::BootstrapMode::STANDARD);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this);

    if (this->loaded) {
        OPENFHE_THROW("Context is already loaded");
    }

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);

    this->slots_bootstrap.push_back(slots);

    // Generate both key sets so the CPU fallback and the GPU pipeline each work:
    //  1. OpenFHE's own bootstrap keys (rotation keys, conjugation key and, for
    //     SPARSE_ENCAPSULATED, the CPU-side sparse-switch keys at automorphism slots 2N-2/2N-4).
    //     The CPU-fallback EvalBootstrap reads exactly those two slots (KeySwitchSparse via
    //     .at(2N-4), KeySwitchInPlace via .at(2N-2)), so without them a CPU bootstrap after this
    //     call throws std::out_of_range ("map::at").
    //  2. FIDESlib::CKKS::GenBootstrapKeys, whose sparse-switch keys live in the disjoint
    //     2N-6/2N-8 slots (BTS_KSPARSE_*_SLOT in RawCiphertext.cu) consumed by the GPU pipeline.
    // The two key sets never collide.
    context->EvalBootstrapKeyGen(skImpl, slots);
    FIDESlib::CKKS::GenBootstrapKeys(skImpl, slots, this->keyDist == fideslib::SPARSE_ENCAPSULATED);
}

// ---- Serialization ----

bool CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(std::ostream& ser, const fideslib::SerType& sertype, const std::string& keyTag) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    bool res;
    switch (sertype) {
    case fideslib::SerType::BINARY:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
            ser, lbcrypto::SerType::BINARY, keyTag);
        break;
    case fideslib::SerType::JSON:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
            ser, lbcrypto::SerType::JSON, keyTag);
        break;
    default: OPENFHE_THROW("Unsupported serialization type");
    }

    return res;
}

bool CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    bool res;
    switch (sertype) {
    case SerType::BINARY:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
            ser, lbcrypto::SerType::BINARY, keyTag);
        break;
    case SerType::JSON:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
            ser, lbcrypto::SerType::JSON, keyTag);
        break;
    default: OPENFHE_THROW("Unsupported serialization type");
    }

    return res;
}

// ---- Deserialization ----

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (!this->devices.empty() && this->loaded) {
        OPENFHE_THROW("DeserializeEvalMultKey must be called before LoadContext");
    }

    bool res;
    switch (sertype) {
    case SerType::BINARY:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
            ser, lbcrypto::SerType::BINARY);
        break;
    case SerType::JSON:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
            ser, lbcrypto::SerType::JSON);
        break;
    default: OPENFHE_THROW("Unsupported serialization type");
    }

    return res;
}

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (!this->devices.empty() && this->loaded) {
        OPENFHE_THROW("DeserializeEvalAutomorphismKey must be called before LoadContext");
    }

    bool res;
    switch (sertype) {
    case SerType::BINARY:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
            ser, lbcrypto::SerType::BINARY);
        break;
    case SerType::JSON:
        res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
            ser, lbcrypto::SerType::JSON);
        break;
    default: OPENFHE_THROW("Unsupported serialization type");
    }

    return res;
}

// ---- Encoding ----

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
    size_t noiseScaleDeg,
    uint32_t level,
    const std::shared_ptr<void> params,
    uint32_t slots) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto pt = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

    Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
    plaintext->cpu = std::make_any<lbcrypto::Plaintext>(pt);
    plaintext->loaded = false;

    if (this->devices.empty() || !this->auto_load_plaintexts) {
        return plaintext;
    }

    this->LoadPlaintext(plaintext);

    return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    auto pt = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

    Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
    plaintext->cpu = std::make_any<lbcrypto::Plaintext>(pt);
    plaintext->loaded = false;

    if (this->devices.empty() || !this->auto_load_plaintexts) {
        return plaintext;
    }

    this->LoadPlaintext(plaintext);

    return plaintext;
}

// ---- Encryption ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
    const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

    auto ct = context->Encrypt(pkImpl, ptImpl);
    Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
    ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

    if (this->devices.empty() || !this->auto_load_ciphertexts) {
        return ciphertext;
    }

    this->LoadCiphertext(ciphertext);

    return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return Encrypt(pt, pk);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    const auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
    const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

    auto ct = context->Encrypt(skImpl, ptImpl);
    Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
    ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

    if (this->devices.empty() || !this->auto_load_ciphertexts) {
        return ciphertext;
    }

    this->LoadCiphertext(ciphertext);

    return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return Encrypt(pt, sk);
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (pt == nullptr) {
        OPENFHE_THROW("Plaintext pointer is null");
    }

    auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

    if (ct->loaded) {
        EnsureMutableCpuCiphertext(ct);
    }

    auto& ct_cpu = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

    // Copy ciphertext to CPU if needed.
    if (ct->loaded) {
        auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
        FIDESlib::CKKS::RawCipherText raw_ct;
        ct_gpu->store(raw_ct);

        // Check if CPU ciphertext has enough levels to hold GPU ciphertext.
        size_t cpu_levels = ct_cpu->GetElements()[0].GetAllElements().size();
        size_t gpu_levels = raw_ct.numRes;

        if (cpu_levels < gpu_levels) {
            // Create a fresh ciphertext at the top level with enough space
            std::vector<double> dummy(1, 0.0);
            auto pt_dummy = context->MakeCKKSPackedPlaintext(dummy, 1, this->multiplicative_depth - ct_gpu->getLevel());
            auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
            ct_cpu = context->Encrypt(skImpl, pt_dummy);
        }

        // Overwrite cpu ct with the data from GPU.
        FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
    }

    auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
    lbcrypto::Plaintext ptImpl;
    auto res = context->Decrypt(skImpl, ct_cpu, &ptImpl);

    if (pt->get() != nullptr) {

        if ((*pt)->loaded && !this->devices.empty()) {
            OPENFHE_THROW("Inconsistent state: Plaintext is marked as loaded but no devices are available");
        }
        if ((*pt)->loaded && !this->EvictDevicePlaintext((*pt)->gpu)) {
            OPENFHE_THROW("Plaintext eviction error: could not evict Plaintext from device");
        }

        (*pt)->cpu = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
        (*pt)->loaded = false;
        (*pt)->gpu = 0;
    } else {
        *pt = std::make_shared<PlaintextImpl>();
        (*pt)->cpu = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
        (*pt)->loaded = false;
        (*pt)->gpu = 0;
    }

    DecryptResult result{};
    result.isValid = res.isValid;
    result.messageLength = res.messageLength;
    return result;
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return Decrypt(ct, sk, pt);
}

// ---- Operations ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalNegate(const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalNegate(ctImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

    // Exact negation (multiply each limb by q_i - 1), matching OpenFHE's EvalNegate:
    // no noise-degree or scaling-factor change, unlike multScalar(-1.0).
    res_gpu->negate();
    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalNegateInPlace(Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        context->EvalNegateInPlace(ctImpl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct);

    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

    // Exact negation (multiply each limb by q_i - 1), matching OpenFHE's EvalNegate.
    ct_gpu->negate();
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ct1Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalAdd(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->add(*ct2_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto ct = context->EvalAdd(ctImpl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
    this->LoadPlaintext(pt);

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->addPt(*pt_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalAdd(ctImpl, scalar);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->addScalar(scalar);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalAdd(ct, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        context->EvalAddInPlace(ct1Impl, ct2Impl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->add(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        context->EvalAddInPlace(ct1Impl, ptImpl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);
    this->LoadPlaintext(pt);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->addPt(*pt_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    EvalAddInPlace(ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        context->EvalAddInPlace(ct1Impl, scalar);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    res_gpu->addScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    EvalAddInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Mutable contract: operands may be adjusted in place (rescaled / level-dropped),
    // matching OpenFHE's EvalAddMutable. Wire to the GPU addMutable and the OpenFHE
    // mutable op so CPU and GPU mutate the same operands and skip the operand copy
    // the immutable forms pay when adjustment is needed.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalAddMutable(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->addMutable(*ct2_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this);

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto res = context->EvalAddMutable(ctImpl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
        return ciphertext;
    }

    this->LoadCiphertext(ct);
    this->LoadPlaintext(pt);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->addPtMutable(*pt_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalAddMutable(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        context->EvalAddMutableInPlace(ct1Impl, ct2Impl);
        return;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->addMutable(*ct2_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (ciphertexts.empty()) {
        OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
    }

    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
        ctImpls.reserve(ciphertexts.size());
        for (const auto& ct : ciphertexts) {
            // Null entries pass through as null impls; the CPU EvalAddMany skips them.
            ctImpls.push_back(ct ? std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu) : lbcrypto::Ciphertext<lbcrypto::DCRTPoly>());
        }
        auto ct = context->EvalAddMany(ctImpls);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.

    // Null entries are skipped, mirroring the CPU implementation.
    size_t first = 0;
    while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
        ++first;
    }
    if (first == ciphertexts.size()) {
        OPENFHE_THROW("EvalAddMany: input ciphertext vector has no non-null entries");
    }
    size_t second = first + 1;
    while (second < ciphertexts.size() && ciphertexts[second] == nullptr) {
        ++second;
    }

    for (const auto& ct : ciphertexts) {
        if (ct != nullptr) {
            this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
        }
    }

    if (second == ciphertexts.size()) {
        // Single non-null entry: independent copy, like the CPU implementation's clone.
        return std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertexts[first]);
    }

    // Left fold, same association as the CPU EvalAddMany: one fresh result, then
    // in-place accumulation.
    Ciphertext<DCRTPoly> result = this->EvalAdd(ciphertexts[first], ciphertexts[second]);
    for (size_t i = second + 1; i < ciphertexts.size(); ++i) {
        if (ciphertexts[i] != nullptr) {
            this->EvalAddInPlace(result, ciphertexts[i]);
        }
    }

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (ciphertexts.empty()) {
        OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
    }

    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
        ctImpls.reserve(ciphertexts.size());
        for (const auto& ct : ciphertexts) {
            // Null entries pass through as null impls; the CPU EvalAddManyInPlace skips them.
            ctImpls.push_back(ct ? std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu) : lbcrypto::Ciphertext<lbcrypto::DCRTPoly>());
        }
        context->EvalAddManyInPlace(ctImpls);
        // The result lands in slot 0 (the in-place contract); wrap it if slot 0 was null.
        if (ciphertexts[0] == nullptr) {
            ciphertexts[0] = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        }
        ciphertexts[0]->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ctImpls[0]);
        return;
    }

    // GPU path.

    for (const auto& ct : ciphertexts) {
        if (ct != nullptr) {
            this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
        }
    }

    // Serial in-place fold into slot 0 (null entries are skipped), matching the CPU
    // EvalAddManyInPlace association.
    size_t first = 0;
    while (first < ciphertexts.size() && ciphertexts[first] == nullptr) {
        ++first;
    }
    if (first == ciphertexts.size()) {
        OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector has no non-null entries");
    }
    for (size_t i = first + 1; i < ciphertexts.size(); ++i) {
        if (ciphertexts[i] != nullptr) {
            this->EvalAddInPlace(ciphertexts[first], ciphertexts[i]);
        }
    }
    if (first != 0) {
        ciphertexts[0] = ciphertexts[first];
    }
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ct1Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalSub(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->sub(*ct2_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto ct = context->EvalSub(ctImpl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
    this->LoadPlaintext(pt);

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->subPt(*pt_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto ct = context->EvalSub(ptImpl, ctImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
    this->LoadPlaintext(pt);

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    // pt - ct (not ct - pt): exact, noiseless negation (as in OpenFHE's EvalNegate) then add the
    // plaintext; multScalar(-1.0) would instead pay a full scalar-multiply (extra noise, and a
    // possible rescale + level drop under the AUTO scaling techniques).
    res_gpu->negate();
    res_gpu->addPt(*pt_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalSub(ctImpl, scalar);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->addScalar(-scalar);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalSub(scalar, ctImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    // scalar - ct (not ct - scalar): exact, noiseless negation (as in OpenFHE's EvalNegate) then
    // add; multScalar(-1.0) would instead pay a full scalar-multiply (extra noise, and a possible
    // rescale + level drop under the AUTO scaling techniques).
    res_gpu->negate();
    res_gpu->addScalar(scalar);

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        context->EvalSubInPlace(ct1Impl, ct2Impl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->sub(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        context->EvalSubInPlace(ct1Impl, scalar);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    res_gpu->addScalar(-scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        context->EvalSubInPlace(scalar, ct1Impl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    // scalar - ct (not ct - scalar): exact, noiseless negation (as in OpenFHE's EvalNegate) then
    // add; multScalar(-1.0) would instead pay a full scalar-multiply (extra noise, and a possible
    // rescale + level drop under the AUTO scaling techniques).
    res_gpu->negate();
    res_gpu->addScalar(scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalSubMutable(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->subMutable(*ct2_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this);

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto res = context->EvalSubMutable(ctImpl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
        return ciphertext;
    }

    this->LoadCiphertext(ct);
    this->LoadPlaintext(pt);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->subPtMutable(*pt_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    // OpenFHE's EvalSubMutable(pt, ct) negates ct, adds mutably, then re-negates to
    // restore ct. EvalSub(pt, ct) computes the same value on a fresh negation without
    // touching ct, so the observable result matches and no operand is left modified.
    return EvalSub(pt, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        context->EvalSubMutableInPlace(ct1Impl, ct2Impl);
        return;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->subMutable(*ct2_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ct1Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalMult(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->mult(*ct2_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ct1Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
        auto ct = context->EvalMult(ct1Impl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
    this->LoadPlaintext(pt);

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->multPt(*pt_gpu);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalMult(ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ct1Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto ct = context->EvalMult(ct1Impl, scalar);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->multScalar(scalar);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalMult(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
        auto res = context->EvalMult(ct1Impl, ptImpl);
        ct1->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);
    this->LoadPlaintext(pt);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->multPt(*pt_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        context->EvalMultInPlace(ct1Impl, scalar);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct1);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    res_gpu->multScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    EvalMultInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        auto ct = context->EvalMultMutable(ct1Impl, ct2Impl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->multMutable(*ct2_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this);

    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
        auto res = context->EvalMultMutable(ctImpl, ptImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
        return ciphertext;
    }

    this->LoadCiphertext(ct);
    this->LoadPlaintext(pt);
    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
    res_gpu->multPtMutable(*pt_gpu);
    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalMultMutable(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct1);
        EnsureMutableCpuCiphertext(ct2);
        auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
        auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
        context->EvalMultMutableInPlace(ct1Impl, ct2Impl);
        return;
    }

    this->LoadCiphertext(ct1);
    this->LoadCiphertext(ct2);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
    auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
    res_gpu->multMutable(*ct2_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquare(const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalSquare(ctImpl);
        Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return ciphertext;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->square();

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSquareInPlace(Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        context->EvalSquareInPlace(ctImpl);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct);

    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    ct_gpu->square();
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquareMutable(Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return EvalSquare(ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    {
        FIDESlib::CudaNvtxRange r("API_fallback");
        // Fall back to CPU.
        if (this->devices.empty()) {

            auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
            auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
            auto ct = context->EvalRotate(ctImpl, index);
            Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
            result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
            return result;
        }
    }
    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->rotate(index);

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->EvalRotate(ctImpl, index);
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ciphertext);

    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
    ct_gpu->rotate(index);
}

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        return context->EvalFastRotationPrecompute(ctImpl);
    }

    // GPU path not needed.

    return nullptr;
}

#include <core/math/hal/bigintdyn/ubintdyn.h>

Ciphertext<DCRTPoly>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto casted = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
        return result;
    }

    // GPU path. Inefficient for only one rotation.

    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    res_gpu->copy(*ct_gpu);
    res_gpu->rotate((int)index, true);

    return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto casted = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
        return result;
    }

    // GPU path. Inefficient for only one rotation.

    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    // ct_gpu->rotate((int)index, false);
    res_gpu->copy(*ct_gpu);
    res_gpu->rotate((int)index, false);

    return result;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    std::vector<Ciphertext<DCRTPoly>> results;

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto casted = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);

        for (const auto& index : indices) {
            Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
            result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
            results.push_back(result);
        }
        return results;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

    // Create result ciphertexts.
    std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
    std::vector<int32_t> indices_real;
    for (int indice : indices) {
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

        if (indice != 0) {
            indices_real.push_back(indice);
            auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
            results_gpu.push_back(res_gpu.get());
        }
        results.push_back(result);
    }

    ct_gpu->rotate_hoisted(indices_real, results_gpu, false);
    return results;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    std::vector<Ciphertext<DCRTPoly>> results;

    // Fall back to CPU.
    if (this->devices.empty()) {
        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto casted = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);

        for (const auto& index : indices) {
            Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
            result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
            results.push_back(result);
        }
        return results;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

    std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
    std::vector<int32_t> indices_real;
    for (int indice : indices) {
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

        if (indice != 0) {
            indices_real.push_back(indice);
            auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
            results_gpu.push_back(res_gpu.get());
        }
        results.push_back(result);
    }

    ct_gpu->rotate_hoisted(indices_real, results_gpu, true);
    return results;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto ct = context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return result;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto res = context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
        ct->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
}

std::vector<double> CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return FIDESlib::CKKS::get_chebyshev_coefficients(func, a, b, degree);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Rescale(const Ciphertext<DCRTPoly>& ciphertext) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->Rescale(ctImpl);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return result;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
    res_gpu->rescale();

    return result;
}

void CryptoContextImpl<DCRTPoly>::RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
        EnsureMutableCpuCiphertext(ciphertext);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->Rescale(ctImpl);
        ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ciphertext);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
    res_gpu->rescaleInternal();
}

void CryptoContextImpl<DCRTPoly>::SetLevel(Ciphertext<DCRTPoly>& ct, size_t level) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    ct->SetLevel(level);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->EvalBootstrap(ctImpl, numIterations, precision);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return result;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

    CheckBootstrapMode(res_gpu->slots, { fideslib::BootstrapMode::STANDARD, fideslib::BootstrapMode::STC_FIRST }, "EvalBootstrap");
    FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled, numIterations, precision);

    return result;
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->EvalBootstrap(ctImpl, numIterations, precision);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        ciphertext = result;
        return;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));

    CheckBootstrapMode(res_gpu->slots, { fideslib::BootstrapMode::STANDARD, fideslib::BootstrapMode::STC_FIRST }, "EvalBootstrapInPlace");
    FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled, numIterations, precision);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrapStCFirst(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
    // Fall back to CPU.
    if (this->devices.empty()) {

        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
        auto ct = context->EvalBootstrapStCFirst(ctImpl, numIterations, precision);
        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
        return result;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

    CheckBootstrapMode(res_gpu->slots, { fideslib::BootstrapMode::STC_FIRST }, "EvalBootstrapStCFirst");
    FIDESlib::CKKS::BootstrapStCFirst(*res_gpu, res_gpu->slots, /*prescaled=*/false, numIterations, precision);

    return result;
}

constexpr int ACCUMULATE_SUM_RADIX = PARTIAL_SUM_RADIX;

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        auto& ctImpl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ctImpl);

        lbcrypto::FHECKKSRNS::EvalPartialSumInPlace(result_ct, stride, slots, ACCUMULATE_SUM_RADIX);

        Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
        result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(result_ct);
        return result;
    }

    // GPU path.
    this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

    Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

    const int inputSlots = res_gpu->slots;
    FIDESlib::CKKS::Accumulate(*res_gpu, ACCUMULATE_SUM_RADIX, stride, slots);
    res_gpu->slots = inputSlots;

    return result;
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });

    if (this->devices.empty()) {
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        lbcrypto::FHECKKSRNS::EvalPartialSumInPlace(ctImpl, stride, slots, ACCUMULATE_SUM_RADIX);
        return;
    }

    // GPU path.
    this->LoadCiphertext(ct);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

    const int inputSlots = res_gpu->slots;
    FIDESlib::CKKS::Accumulate(*res_gpu, ACCUMULATE_SUM_RADIX, stride, slots);
    res_gpu->slots = inputSlots;
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {

    if (this->devices.empty()) {
        EnsureMutableCpuCiphertext(ct);
        auto& ctImpl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

        // Same lazy radix fold as the GPU path below: the start-offset EvalPartialSumInPlace in
        // the patched OpenFHE reference (deps/fideslib-1.6.1-compile.patch) keeps the CPU and GPU
        // results bit-exact (previously this was an eager doubling loop, which is never bit-exact
        // against the GPU's lazy fold).
        lbcrypto::FHECKKSRNS::EvalPartialSumInPlace(ctImpl, stride, slots, ACCUMULATE_SUM_RADIX, start);

        return;
    }

    // GPU path.
    this->LoadCiphertext(ct);

    auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

    // Same radix as the plain variants (never a literal), with the same slots save/restore:
    // the unified `Accumulate` is metadata-neutral (no trailing slots rewrite, O10).
    const int inputSlots = res_gpu->slots;
    FIDESlib::CKKS::Accumulate(*res_gpu, ACCUMULATE_SUM_RADIX, stride, slots, start);
    res_gpu->slots = inputSlots;
}

void CryptoContextImpl<DCRTPoly>::ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
    int gStep,
    int bStep,
    const std::vector<Plaintext>& pts,
    const std::vector<int>& indexes,
    int stride,
    int rowSize) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    if (this->devices.empty()) {
        OPENFHE_THROW("Not implemented for CPU path");
    }

    // GPU path.
    this->LoadCiphertext(ct);
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
    pts_gpu.reserve(pts.size());
    for (const auto& pt : pts) {
        this->LoadPlaintext(const_cast<Plaintext&>(pt));
        auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
        pts_gpu.push_back(pt_gpu.get());
    }

    if (rowSize == 0) {
        rowSize = bStep * gStep;
    }

    FIDESlib::CKKS::ConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, indexes, gStep);
}

void CryptoContextImpl<DCRTPoly>::SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
    int gStep,
    int bStep,
    const std::vector<Plaintext>& pts,
    Plaintext& mask,
    const std::vector<int>& indexes,
    int stride,
    int maskRotationStride,
    int rowSize) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    PlaintextCacheHold plaintext_hold(*this); // no plaintext this op fetches may be evicted mid-op

    if (this->devices.empty()) {
        OPENFHE_THROW("Not implemented for CPU path");
    }

    // GPU path.
    this->LoadCiphertext(ct);
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
    std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
    pts_gpu.reserve(pts.size());
    for (const auto& pt : pts) {
        this->LoadPlaintext(const_cast<Plaintext&>(pt));
        auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
        pts_gpu.push_back(pt_gpu.get());
    }

    // Load mask
    this->LoadPlaintext(mask);
    auto mask_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(mask->gpu));

    if (rowSize == 0) {
        rowSize = bStep * gStep;
    }

    FIDESlib::CKKS::SpecialConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, *mask_gpu, stride, maskRotationStride, indexes, gStep);
}

// ---- Copy helpers ----

uint32_t CryptoContextImpl<DCRTPoly>::CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (!ct.loaded) {
        OPENFHE_THROW("Ciphertext not loaded to any device");
    }

    auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct.gpu));
    // RNSPoly::copy()'s cross-stream wait (s.wait(partition.getS())) does not observe per-limb
    // streams that a preceding op (e.g. multPt's NTT step) left work on without folding it back
    // into the partition's main stream in time -- draining ct_gpu's own streams here, the same
    // way Ciphertext::store() does before a Decrypt/serialize read, closes that race. Without
    // this, chaining an op straight off a ciphertext that just came out of a plaintext multiply
    // (no intervening Decrypt/Synchronize) can read/copy it while writes are still in flight.
    ct_gpu->c0.sync();
    ct_gpu->c1.sync();
    auto new_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu);
    new_ct->copy(*ct_gpu);
    uint32_t handle = this->RegisterDeviceCiphertext(std::move(new_ct));
    return handle;
}

// ---- Map Handling ----

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDevicePlaintext(std::shared_ptr<void>&& p) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->devices.empty()) {
        OPENFHE_THROW("No devices available to register plaintext");
    }
    // device_plaintexts_mutex->lock();
    uint32_t handle = next_gpu_handle++;
    device_plaintexts.emplace(handle, std::move(p));
    // device_plaintexts_mutex->unlock();
    return handle;
}

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDeviceCiphertext(std::shared_ptr<void>&& c) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->devices.empty()) {
        OPENFHE_THROW("No devices available to register ciphertext");
    }
    // device_ciphertexts_mutex->lock();
    uint32_t handle = next_gpu_handle++;
    // Measure before the move; every ciphertext registered here is a FIDESlib::CKKS::Ciphertext.
    const size_t bytes = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(c)->limbBytes();
    device_ciphertexts.emplace(handle, std::move(c));
    // device_ciphertexts_mutex->unlock();
    // Tracked, but deliberately NOT an eviction point: an operation registers its result (or
    // a whole batch, in the hoisted rotations) while holding raw pointers to device
    // ciphertexts it has already fetched. The budget is enforced in LoadCiphertext() instead,
    // so it overshoots by one operation's working set.
    this->TrackCiphertext(handle, bytes);
    return handle;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDevicePlaintext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    // device_plaintexts_mutex->lock_shared();
    auto& it = device_plaintexts.at(handle);
    // device_plaintexts_mutex->unlock_shared();

    // Every operation reaches a device plaintext through this one choke point, so the cache's
    // recency order is maintained here for all of them at once.
    this->TouchPlaintext(handle);
    return it;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDeviceCiphertext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    // device_ciphertexts_mutex->lock_shared();
    auto& it = device_ciphertexts.at(handle);
    // device_ciphertexts_mutex->unlock_shared();

    // Every caller (receiver or read-only operand alike) fetches the GPU ciphertext through
    // this single choke point, so restoring an offloaded ciphertext here -- rather than at each
    // of the ~40 call sites below -- transparently covers all of them.
    auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(it);
    if (ct_gpu->isOffloaded()) {
        ct_gpu->reload();
    }
    // Recency and size bookkeeping for the cache (the size is re-read here rather than
    // remembered from the load: key-switching grows and frees the special limbs). Not an
    // eviction point either, for the same reason as RegisterDeviceCiphertext().
    this->TouchCiphertext(handle, ct_gpu->limbBytes());
    return it;
}

bool CryptoContextImpl<DCRTPoly>::EvictDevicePlaintext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    // Every path out of the registry (cache eviction, Decrypt into a plaintext, ~PlaintextImpl,
    // an explicit unload) goes through here, so the cache bookkeeping is dropped here too.
    this->UntrackPlaintext(handle);
    // device_plaintexts_mutex->lock();
    auto result = device_plaintexts.erase(handle) > 0;
    // device_plaintexts_mutex->unlock();
    return result;
}

bool CryptoContextImpl<DCRTPoly>::EvictDeviceCiphertext(uint32_t handle) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    this->UntrackCiphertext(handle);
    // device_ciphertexts_mutex->lock();
    auto result = device_ciphertexts.erase(handle) > 0;
    // device_ciphertexts_mutex->unlock();
    return result;
}

void CryptoContextImpl<DCRTPoly>::Synchronize() const {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    if (this->devices.empty() || !this->loaded) {
        return;
    }
    for (const auto& device : this->devices) {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        CudaCheckErrorModNoSync;
    }
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
    FIDESlib::CudaNvtxRange r("API" + std::string{ sc::current().function_name() });
    return FIDESlib::CKKS::GetConvolutionTransformRotationIndices(rowSize, bStep, stride, gStep);
}

} // namespace fideslib