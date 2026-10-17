//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include "Plaintext.cuh"
#include <vector>

namespace FIDESlib::CKKS {

/** Host copy of a group of plaintexts: every limb of every plaintext back to back in one
 * buffer, page-locked when the driver allows it so that a reload is a straight DMA at full
 * PCIe bandwidth (and falling back to ordinary memory otherwise). */
class BootSnapshot {
  public:
    struct Entry {
        size_t first_row = 0;         ///< index of the plaintext's first limb in the buffer
        std::vector<uint64_t> moduli; ///< one per limb, as RawPlainText::moduli
        double noise = 0;
        int noise_level = 1;
        int slots = 0;
    };

    BootSnapshot() = default;
    /** Decimate the full `n`-word raw limbs to upstream's 2*slots layout and keep
     * them in one pinned buffer. Rows are padded to the widest entry in this group. */
    BootSnapshot(const std::vector<RawPlainText>& raws, size_t n);

    [[nodiscard]] bool empty() const {
        return entries.empty();
    }

    [[nodiscard]] bool pinned() const {
        return buffer.pinned();
    }

    /** Host pointers to the compressed limbs of entry `i`, in order (2*entry(i).slots words). */
    [[nodiscard]] std::vector<const uint64_t*> rows(size_t i) const;

    [[nodiscard]] size_t rowWords() const {
        return buffer.rowWords();
    }

    [[nodiscard]] size_t size() const {
        return entries.size();
    }

    [[nodiscard]] const Entry& entry(size_t i) const {
        return entries.at(i);
    }

  private:
    std::vector<Entry> entries;
    HostRows buffer;
};

/** VRAM-cache state of one group of linear-transform plaintexts that are always consumed
 * together: one CoeffsToSlots/SlotsToCoeffs stage, or LT.A / LT.invA. The group is the unit
 * the bootstrap cache loads and evicts (see ContextData::SetBootstrapCache()). */
struct BootCacheBlock {
    /** Host-side copy of every plaintext of the group, in order. Kept only when the group was
     * built under a finite budget; a group without it can never be evicted. */
    BootSnapshot snapshot;
    bool resident = true;
    /** Device bytes the group holds while resident (recorded when it is built). */
    size_t bytes = 0;
    /** Value of ContextData::bootstrap_cache_clock at the group's last use. */
    uint64_t last_use = 0;

    [[nodiscard]] bool hasSnapshot() const {
        return !snapshot.empty();
    }
};

class BootstrapPrecomputation {
  public:
    struct {
        int slots = -1;
        int bStep = -1;
        std::vector<Plaintext> A;
        std::vector<Plaintext> invA;
        BootCacheBlock cacheA;
        BootCacheBlock cacheInvA;
    } LT;

    struct LTstep {
        int slots = -1;
        int bStep = -1;
        int gStep = -1;
        std::vector<Plaintext> A;
        std::vector<int> rotIn;
        std::vector<int> rotOut;
        BootCacheBlock cache;
    };

    std::vector<LTstep> StC;
    std::vector<LTstep> CtS;
    int accumulate_bStep = 4;
    uint32_t correctionFactor;
    bool sparse_encaps{ false };
    bool btsfirstboot{ false };
    /// CKKSDataType == COMPLEX for this precomputation. The reference's evaluators branch on the
    /// datatype in exactly one place that touches the bootstrap: the sparsely packed SlotsToCoeffs
    /// fold is applied only for REAL. COMPLEX sparse/LT precomputations carry this so the GPU skips it.
    bool isComplex{ false };
    int lvlbDec{ -1 };

    // Bookkeeping flags for the shared per-slot bootstrap precomputation. They are never set here
    // (this branch has no functional setup path) and only feed the inert mode guards in Bootstrap.cu.
    bool isFBT{ false };
    bool isFEFunc{ false };

    std::weak_ptr<ContextData> sparse_context;

    /** Calls f(block, plaintexts) for every cached group of plaintexts. */
    template <typename F> void forEachBlock(F&& f) {
        forEachBlockOf(*this, f);
    }

    template <typename F> void forEachBlock(F&& f) const {
        forEachBlockOf(*this, f);
    }

  private:
    template <typename Self, typename F> static void forEachBlockOf(Self& self, F& f) {
        if (!self.LT.A.empty())
            f(self.LT.cacheA, self.LT.A);
        if (!self.LT.invA.empty())
            f(self.LT.cacheInvA, self.LT.invA);
        for (auto* steps : { &self.CtS, &self.StC })
            for (auto& step : *steps)
                f(step.cache, step.A);
    }
};
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
