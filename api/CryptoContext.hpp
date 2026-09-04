#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <complex>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "Definitions.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"

namespace fideslib {

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	void SetDevices(const std::vector<int>& devices);

	// ---- Rotation-key VRAM cache ----

	/// @brief Cap the VRAM spent on rotation keys to `bytes`, offloading the least recently
	/// used ones to host RAM and reloading them on demand. SIZE_MAX (the default) keeps every
	/// rotation key permanently resident.
	///
	/// Call it BEFORE LoadContext(): rotation keys can only be offloaded if they were created
	/// while a finite budget was set (they then keep the host-RAM snapshot that offload/reload
	/// round-trips through, and cost no VRAM until first used). Keys already resident when the
	/// budget is set have no snapshot and stay pinned; setting the budget after LoadContext()
	/// therefore only bounds keys added later (e.g. by EvalBootstrapKeyGen).
	///
	/// The budget is soft: operations that need several keys at once (hoisted rotation,
	/// bootstrap's linear transforms) transiently overshoot it and the cache re-shrinks on the
	/// next load.
	void SetRotationKeyCache(size_t bytes);
	/// @brief The current rotation-key VRAM budget in bytes (SIZE_MAX = unlimited).
	size_t GetRotationKeyCache() const;
	/// @brief Offload rotation keys to host RAM now, without waiting for eviction. An empty
	/// `indexes` offloads all of them. Keys without a host snapshot, and pinned keys, are
	/// skipped. Requires the context to be loaded.
	void OffloadRotationKeys(const std::vector<int>& indexes = {});
	/// @brief Pin (or unpin) a rotation key so the cache never evicts it. Requires the context
	/// to be loaded.
	void PinRotationKey(int index, bool pin = true);
	/// @brief Whether the rotation key for `index` currently holds its VRAM limbs.
	bool IsRotationKeyResident(int index) const;
	/// @brief VRAM bytes currently spent on resident rotation keys.
	size_t GetRotationKeyCacheResidentBytes() const;

	// ---- Plaintext VRAM cache ----

	/// @brief Cap the VRAM spent on device plaintexts to `bytes`, dropping the least recently
	/// used ones and re-uploading them on demand. SIZE_MAX (the default) keeps every plaintext
	/// on the device until its Plaintext is destroyed.
	///
	/// Unlike the rotation-key cache this can be called at any time, and needs no snapshot:
	/// the host-side encoding a plaintext was built from is kept by its Plaintext anyway, so
	/// eviction is a plain free and a miss costs one host->device upload of an unchanged
	/// encoding (plaintexts are read-only in every operation that takes one). Every operation
	/// re-uploads what it needs through LoadPlaintext(), so eviction is transparent.
	///
	/// The budget is soft in three ways: a plaintext's size is only known once it is built, so
	/// a load overshoots by that one plaintext before the cache re-shrinks; an operation
	/// holding several plaintexts at once (the convolution transforms) keeps all of them
	/// resident until it returns; and a budget smaller than a single plaintext still keeps the
	/// one in use resident. Pinned plaintexts are never evicted.
	///
	/// Only plaintexts registered through LoadPlaintext() are cached -- not the plaintexts
	/// inside the bootstrapping precomputation, which belong to the GPU context.
	void SetPlaintextCache(size_t bytes);
	/// @brief The current plaintext VRAM budget in bytes (SIZE_MAX = unlimited).
	size_t GetPlaintextCache() const;
	/// @brief VRAM bytes currently spent on device plaintexts. Tracked whether or not a
	/// budget is set, so it also answers "how much VRAM are my plaintexts holding?".
	size_t GetPlaintextCacheResidentBytes() const;
	/// @brief Drop every unpinned device plaintext now, without waiting for eviction. The
	/// plaintexts stay usable: the next operation re-uploads what it needs.
	void OffloadPlaintexts();
	/// @brief Pin (or unpin) a plaintext so the cache never evicts it, loading it to the
	/// device first if needed. The pin is tied to the current device copy: dropping it
	/// explicitly (Decrypt into it, or destroying it) also drops the pin.
	void PinPlaintext(Plaintext& pt, bool pin = true);

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load.
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	// ---- Offload to host RAM ----

	/// @brief Evict a loaded ciphertext's GPU limbs to host RAM, freeing VRAM. The ciphertext
	/// stays registered (its handle remains valid) so it can be transparently restored on next
	/// use. A no-op if the ciphertext isn't currently loaded on a device, or handle is 0.
	void OffloadCiphertext(uint32_t handle);
	/// @brief Restore a ciphertext previously evicted by OffloadCiphertext(). A no-op if the
	/// ciphertext isn't currently offloaded, or handle is 0.
	void ReloadCiphertext(uint32_t handle);
	/// @brief Whether the ciphertext behind `handle` is currently offloaded to host RAM.
	bool IsCiphertextOffloaded(uint32_t handle) const;
	/// @brief Give idle VRAM back to the CUDA driver after offloading a batch of
	/// ciphertexts. OffloadCiphertext() only returns freed limbs to FIDESlib's own
	/// process-local allocator pool (so they stay cheap to reuse for more FIDESlib
	/// ciphertexts/ops), not to the driver, so `nvidia-smi`/`cudaMemGetInfo` won't show
	/// a drop from OffloadCiphertext() alone. Call this once after offloading everything
	/// you want the memory back from, e.g. before allocating unrelated GPU memory for
	/// other work. A no-op if the context isn't loaded to any device.
	void TrimGPUMemoryPool();

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget = { 5, 4 },
	  std::vector<uint32_t> dim1									 = { 0, 0 },
	  uint32_t slots												 = 0,
	  uint32_t correctionFactor										 = 0,
	  bool precompute												 = true,
	  bool btsfirstboot												 = false);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);

	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);

	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	std::any cpu;
	std::any gpu;
	/// @brief Whether the context has been loaded to the devices.
	bool loaded = false;
	/// @brief List of devices the context is loaded on.
	std::vector<int> devices = { 0 };
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Bootstrap slots available.
	std::vector<uint32_t> slots_bootstrap;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;
	/// @brief Rotation-key VRAM budget in bytes, applied to the GPU context in LoadContext();
	/// SIZE_MAX means unlimited. See SetRotationKeyCache().
	size_t rotation_key_cache_bytes = SIZE_MAX;

	// ---- Copy helpers ----

	uint32_t CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct);

	// --- Map Handling ----

	/// @brief  Registry of plaintexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_plaintexts;
	/// @brief  Registry of ciphertexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_ciphertexts;
	/// @brief Next available handle for GPU objects. Zero is reserved as a null handle.
	uint32_t next_gpu_handle = 1;

	/// @brief Plaintext VRAM budget in bytes; SIZE_MAX means unlimited. See SetPlaintextCache().
	size_t plaintext_cache_bytes = SIZE_MAX;
	/// @brief VRAM bytes held by the plaintexts currently in `device_plaintexts`.
	size_t plaintext_resident_bytes = 0;
	/// @brief Cache bookkeeping for one device plaintext.
	struct PlaintextCacheEntry {
		/// @brief The api-level plaintext owning this handle, so eviction can mark it unloaded.
		std::weak_ptr<PlaintextImpl> owner;
		/// @brief VRAM bytes the device plaintext held when it was loaded.
		size_t bytes = 0;
		/// @brief This handle's position in `plaintext_lru`.
		std::list<uint32_t>::iterator lru;
	};
	std::unordered_map<uint32_t, PlaintextCacheEntry> plaintext_cache;
	/// @brief Handles by recency of use, most recent first; eviction takes from the back.
	std::list<uint32_t> plaintext_lru;
	/// @brief Handles the cache must never evict. See PinPlaintext().
	std::unordered_set<uint32_t> plaintext_pinned;
	/// @brief Depth of active PlaintextCacheHold scopes; eviction is deferred while > 0.
	int plaintext_cache_hold = 0;

	/// @brief Start tracking a freshly loaded device plaintext as the most recently used one.
	void TrackPlaintext(uint32_t handle, const Plaintext& pt, size_t bytes);
	/// @brief Stop tracking a device plaintext that is about to leave `device_plaintexts`.
	void UntrackPlaintext(uint32_t handle);
	/// @brief Mark a device plaintext as the most recently used one.
	void TouchPlaintext(uint32_t handle);
	/// @brief Drop one device plaintext, marking its Plaintext unloaded so the next operation
	/// re-uploads it. Returns false if the handle isn't tracked.
	bool OffloadPlaintextHandle(uint32_t handle);
	/// @brief Evict least-recently-used plaintexts until the budget is met, never touching
	/// `protect`, a pinned handle, or anything at all while a PlaintextCacheHold is active.
	void EnforcePlaintextBudget(uint32_t protect = 0);

	uint32_t RegisterDevicePlaintext(std::shared_ptr<void>&& p);
	uint32_t RegisterDeviceCiphertext(std::shared_ptr<void>&& c);
	std::shared_ptr<void>& GetDevicePlaintext(uint32_t handle);
	std::shared_ptr<void>& GetDeviceCiphertext(uint32_t handle);
	bool EvictDevicePlaintext(uint32_t handle);
	bool EvictDeviceCiphertext(uint32_t handle);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);
};

/// @brief RAII guard that defers plaintext cache eviction to the end of the scope, and runs it
/// there. Every operation that fetches device plaintexts takes one: an operation may hold
/// several of them (the convolution transforms keep raw pointers to a whole batch), and an
/// eviction triggered by loading one of them would free a plaintext whose consuming kernels
/// have not been enqueued yet. Loads still happen inside the scope, they just never evict, so
/// the budget is met again as soon as the operation returns.
struct PlaintextCacheHold {
	CryptoContextImpl<DCRTPoly>& cc;
	explicit PlaintextCacheHold(CryptoContextImpl<DCRTPoly>& cc_) : cc(cc_) { ++cc.plaintext_cache_hold; }
	~PlaintextCacheHold() {
		if (--cc.plaintext_cache_hold == 0) {
			try {
				cc.EnforcePlaintextBudget();
			} catch (...) {
				// A destructor must not throw; the budget is soft and gets enforced again on
				// the next load anyway.
			}
		}
	}
	PlaintextCacheHold(const PlaintextCacheHold&)			 = delete;
	PlaintextCacheHold& operator=(const PlaintextCacheHold&) = delete;
};

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP