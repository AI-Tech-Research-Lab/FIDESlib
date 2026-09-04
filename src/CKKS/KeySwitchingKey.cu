//
// Created by carlosad on 26/09/24.
//

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include <source_location>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
// constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
// constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk, const bool lazy) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc);
	keyID = rkk.keyid;

	if (lazy) {
		// Rotation-key VRAM cache active: keep only the host snapshot. The limbs are
		// generated on first use (ensureResident), so creating a full set of rotation
		// keys never spikes VRAM by sum(key sizes).
		snapshot		 = rkk;
		has_snapshot = true;
		resident		 = false;
	} else {
		loadLimbs(rkk);
		resident = true;
	}
	limb_bytes = computeLimbBytes();
}

void KeySwitchingKey::loadLimbs(RawKeySwitchKey& rkk) {
	a.generateDecompAndDigit(true);
	b.generateDecompAndDigit(true);
	if (cc->GPUid.size() > 1) {
		a.grow(cc->L, false, true);
		b.grow(cc->L, false, true);
	}
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

	cudaDeviceSynchronize();
}

size_t KeySwitchingKey::computeLimbBytes() const {
	size_t bytes		 = 0;
	const bool multi_gpu = cc->GPUid.size() > 1;
	for (size_t g = 0; g < cc->GPUid.size(); ++g) {
		for (const auto& grp : cc->decompMeta.at(g))
			for (const auto& rec : grp)
				bytes += (size_t)cc->N * (rec.type == U32 ? 4 : 8);
		for (const auto& grp : cc->digitMeta.at(g))
			for (const auto& rec : grp)
				bytes += (size_t)cc->N * (rec.type == U32 ? 4 : 8);
		if (multi_gpu) {
			// grow(L, false, true) allocates the regular limbs too (multi-GPU keys only).
			for (const auto& rec : cc->meta.at(g))
				bytes += (size_t)cc->N * (rec.type == U32 ? 4 : 8);
		}
	}
	return 2 * bytes; // a and b have identical shape
}

void KeySwitchingKey::offload() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	if (!resident) {
		return;
	}
	assert(has_snapshot && "Key without a host snapshot cannot be offloaded");
	CKKS::SetCurrentContext(cc);
	// KSK limbs are read by kernels enqueued on other partitions'/auxiliary streams
	// (dotKSK waits on the ksk's stream but launches elsewhere, fused hoisting collects
	// keys before launching). A full drain is the only way to prove the frees are safe;
	// offload happens at most once per key per op, so this is acceptable.
	for (int dev : cc->GPUid) {
		cudaSetDevice(dev);
		cudaDeviceSynchronize();
	}
	for (RNSPoly* poly : { &a, &b }) {
		for (auto& g : poly->GPU)
			g.freeDecompDigitLimbs();
		if (cc->GPUid.size() > 1) {
			// Regular limbs allocated by grow() in the multi-GPU path.
			poly->freeGPU();
		}
	}
	resident = false;
}

void KeySwitchingKey::ensureResident() {
	if (resident) {
		return;
	}
	assert(has_snapshot && "Key without a host snapshot cannot be reloaded");
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc);
	loadLimbs(snapshot);
	resident = true;
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
: my_range(loc, LIFETIME), keyID(""), cc((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)),
  a(*cc, -1, false, true), b(*cc, -1, false, true) {
	CudaNvtxStop();
	/*
	if (cc.GPUid.size() > 1) {
		for (int j = 0; j < cc.dnum; ++j) {
			mgpu_a.emplace_back(cc, -1);
			mgpu_b.emplace_back(cc, -1);
		}
	}
	 */
}
} // namespace FIDESlib::CKKS
