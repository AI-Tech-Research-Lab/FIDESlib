//
// Created by carlosad on 26/09/24.
//

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include <cstring>
#include <set>
#include <source_location>
#include <stdexcept>
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
		size_t rows = 0, n = 0;
		for (const auto& poly : rkk.r_key)
			for (const auto& digit : poly)
				for (const auto& limb : digit) {
					n = limb.size();
					++rows;
				}
		snapshot.rows	= HostRows(rows, n);
		snapshot.moduli = rkk.r_key_moduli;
		snapshot.first_row.assign(rkk.r_key.size(), {});
		size_t row = 0;
		for (size_t p = 0; p < rkk.r_key.size(); ++p) {
			for (const auto& digit : rkk.r_key[p]) {
				snapshot.first_row[p].push_back(row);
				for (const auto& limb : digit) {
					if (limb.size() != n)
						throw std::invalid_argument("KeySwitchingKey: limbs of different lengths");
					std::memcpy(snapshot.rows.row(row++), limb.data(), n * sizeof(uint64_t));
				}
			}
		}
		has_snapshot = true;
		resident	 = false;
	} else {
		loadLimbs(rkk);
		resident = true;
	}
	limb_bytes = computeLimbBytes();
}

void KeySwitchingKey::allocateLimbs() {
	a.generateDecompAndDigit(true);
	b.generateDecompAndDigit(true);
	if (cc->GPUid.size() > 1) {
		a.grow(cc->L, false, true);
		b.grow(cc->L, false, true);
	}
}

void KeySwitchingKey::synchronizeDevices() {
	int current;
	cudaGetDevice(&current);
	for (int dev : std::set<int>(cc->GPUid.begin(), cc->GPUid.end())) {
		cudaSetDevice(dev);
		cudaDeviceSynchronize();
	}
	cudaSetDevice(current);
}

void KeySwitchingKey::loadLimbs(RawKeySwitchKey& rkk) {
	allocateLimbs();
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

	synchronizeDevices();
}

void KeySwitchingKey::loadLimbsFromSnapshot() {
	allocateLimbs();
	RNSPoly* polys[2] = { &a, &b };
	for (size_t p = 0; p < 2; ++p) {
		std::vector<std::vector<const uint64_t*>> rows(snapshot.moduli.at(p).size());
		for (size_t d = 0; d < rows.size(); ++d)
			for (size_t l = 0; l < snapshot.moduli[p][d].size(); ++l)
				rows[d].push_back(snapshot.rows.row(snapshot.first_row[p][d] + l));
		polys[p]->loadDecompDigit(rows, snapshot.rows.rowWords(), snapshot.moduli[p]);
	}
	// The copies read straight from the pinned snapshot: wait for them before returning.
	synchronizeDevices();
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
	loadLimbsFromSnapshot();
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
