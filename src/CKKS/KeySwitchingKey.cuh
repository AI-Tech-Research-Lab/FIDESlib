//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cinttypes>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {

class KeySwitchingKey {
	static constexpr const char* loc{ "KeySwitchingKey" };
	CudaNvtxRange my_range;

  public:
	KeyHash keyID;
	Context& cc;
	RNSPoly a;
	RNSPoly b;

	/** Host-side (RAM) snapshot of the raw key material. Retained when the key is created
	 * with Initialize(..., lazy = true) (i.e. the rotation-key VRAM cache is active); it is
	 * what offload()/ensureResident() round-trip through. Empty (has_snapshot == false)
	 * otherwise: such keys cannot be offloaded and are treated as pinned by the cache. */
	RawKeySwitchKey snapshot;
	bool resident	 = false;
	bool has_snapshot = false;

	explicit KeySwitchingKey(Context& cc);

	/** Load the key into VRAM from `rkk`. With lazy = true (rotation-key VRAM cache
	 * active), no VRAM is allocated here at all: only a host RAM snapshot is kept and
	 * the limbs are generated on first use via ensureResident(). */
	void Initialize(RawKeySwitchKey& rkk, bool lazy = false);

	[[nodiscard]] bool isResident() const { return resident; }
	[[nodiscard]] bool hasSnapshot() const { return has_snapshot; }

	/** VRAM bytes occupied by this key's DECOMP/DIGIT limbs (plus multi-GPU regular
	 * limbs). Computed from context metadata in Initialize(), valid for resident and
	 * offloaded keys alike. */
	[[nodiscard]] size_t limbBytes() const { return limb_bytes; }

	/** Free every VRAM limb of a and b, keeping the host snapshot for a later
	 * ensureResident(). No-op if already offloaded. Requires has_snapshot.
	 * Synchronizes every device first: KSK limbs are read by kernels enqueued on other
	 * partitions'/auxiliary streams (dotKSK, fused hoisting), so stream-ordered frees
	 * alone cannot prove safety. */
	void offload();

	/** (Re)generate the VRAM limbs from the host snapshot. No-op if resident.
	 * Synchronizes before returning, since callers launch kernels right after. */
	void ensureResident();

  private:
	void loadLimbs(RawKeySwitchKey& rkk);
	[[nodiscard]] size_t computeLimbBytes() const;
	size_t limb_bytes = 0;
};

} // namespace CKKS
} // namespace FIDESlib

#endif // GPUCKKS_KEYSWITCHINGKEY_CUH
