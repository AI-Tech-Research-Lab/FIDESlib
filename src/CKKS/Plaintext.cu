//
// Created by carlosad on 25/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
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

namespace {
// Ring element of one limb in coefficient form: coefficient k of the encoding at position k * gap,
// zero elsewhere. Same residues as OpenFHE's CKKSPackedEncoding::FitToNativeVector followed by the
// optional constant multiplication (DCRTPoly::Times), both of which leave canonical values in [0, q).
template <typename T>
__global__ void scatterReduceCoefficients(T* out, const int64_t* coefficients, const int n, const int gap, const uint64_t q,
    const uint64_t factor, const int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        if (idx % gap != 0) {
            out[idx] = 0;
            continue;
        }
        const int64_t x = coefficients[idx / gap];
        uint64_t r = x >= 0 ? static_cast<uint64_t>(x) % q : (q - (static_cast<uint64_t>(-x) % q)) % q;
        if (factor != 1)
            r = static_cast<uint64_t>(static_cast<unsigned __int128>(r) * factor % q);
        out[idx] = static_cast<T>(r);
        (void)n;
    }
}

// Keep every gap-th word of a full-ring evaluation-form limb: the compressed layout
// RNSPoly::loadConstant(..., compress = true) builds on the host for sparsely packed plaintexts.
template <typename T> __global__ void gatherStrided(T* out, const T* in, const int n, const int gap) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
        out[idx] = in[static_cast<size_t>(idx) * gap];
}

constexpr int kEncodeBlock = 256;

int EncodeGrid(const int n) {
    return std::max(1, std::min((n + kEncodeBlock - 1) / kEncodeBlock, 4096));
}
} // namespace

Plaintext::Plaintext(Context& cc, const RawPlainCoefficients& raw)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(this->cc, -1, false, true) {
    load(raw);
    CudaNvtxStop();
    c0.onSizeChanged = [] { CudaNvtxLifetimeRefresh(Plaintext::loc); };
    CudaNvtxLifetimeRegisterBytes(loc, this, [this] { return this->getMemoryUsage(); });
}

void Plaintext::load(const RawPlainCoefficients& raw) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    const int limbs = static_cast<int>(raw.moduli.size());
    const int n = 2 * raw.slots;
    if (limbs == 0 || limbs > cc.L + 1 || raw.N != cc.N || raw.slots <= 0 || n > cc.N || cc.N % n != 0 ||
        raw.coefficients.size() != static_cast<size_t>(n) || (!raw.factors.empty() && raw.factors.size() != raw.moduli.size()))
        throw std::invalid_argument("Plaintext::load: inconsistent coefficient encoding");
    for (int i = 0; i < limbs; ++i) {
        if (raw.moduli[i] != cc.prime.at(i).p)
            throw std::invalid_argument("Plaintext::load: plaintext moduli do not match the context's primes");
        if (!raw.factors.empty() && raw.factors[i] >= raw.moduli[i])
            throw std::invalid_argument("Plaintext::load: per-limb factors must be reduced");
    }
    NoiseFactor = raw.Noise;
    NoiseLevel = raw.NoiseLevel;
    slots = raw.slots;
    const int gap = cc.N / n;

    // Destination: the same compressed constant layout loadConstant() builds.
    if (c0.getLevel() < limbs - 1)
        c0.grow(limbs - 1, false, true, n);
    else
        c0.dropToLevel(limbs - 1);

    // Scratch full-ring element to run the NTT on (constant limbs carry no NTT workspace).
    RNSPoly full(cc, limbs - 1);
    assert(full.GPU.size() == c0.GPU.size());

    for (auto& g : full.GPU) {
        cudaSetDevice(g.device);
        const int limbSize = static_cast<int>(g.getLimbSize(limbs - 1));
        if (limbSize == 0)
            continue;
        VectorGPU<uint64_t> coefficients(g.s, n, g.device, reinterpret_cast<const uint64_t*>(raw.coefficients.data()));
        for (int j = 0; j < limbSize; ++j) {
            Stream& stream = STREAM(g.limb[j]);
            stream.wait(g.s);
            const int primeid = PRIMEID(g.limb[j]);
            const uint64_t q = cc.prime.at(primeid).p;
            const uint64_t factor = raw.factors.empty() ? 1 : raw.factors[primeid];
            std::visit(
                [&](auto& limb) {
                    scatterReduceCoefficients<<<EncodeGrid(cc.N), kEncodeBlock, 0, stream.ptr()>>>(limb.v.data,
                        reinterpret_cast<const int64_t*>(coefficients.data), n, gap, q, factor, cc.N);
                },
                g.limb[j]);
            g.s.wait(stream);
        }
        coefficients.free(g.s); // stream-ordered after every scatter through g.s
    }

    full.NTT(cc.batch, true); // waits for g.s before, and g.s waits for the NTT after

    for (size_t p = 0; p < full.GPU.size(); ++p) {
        auto& g = full.GPU[p];
        auto& d = c0.GPU[p];
        cudaSetDevice(g.device);
        const int limbSize = static_cast<int>(g.getLimbSize(limbs - 1));
        for (int j = 0; j < limbSize; ++j) {
            assert(PRIMEID(g.limb[j]) == PRIMEID(d.limb[j]));
            Stream& stream = STREAM(d.limb[j]);
            stream.wait(g.s);
            std::visit(
                [&](auto& dst) {
                    using T = std::remove_reference_t<decltype(dst.v.data[0])>;
                    const T* src = std::get<std::is_same_v<T, uint32_t> ? U32 : U64>(g.limb[j]).v.data;
                    gatherStrided<<<EncodeGrid(n), kEncodeBlock, 0, stream.ptr()>>>(dst.v.data, src, n, gap);
                },
                d.limb[j]);
            g.s.wait(stream);
        }
        // The scratch limbs are released on their own streams: order that after the gathers.
        for (int j = 0; j < limbSize; ++j)
            STREAM(g.limb[j]).wait(g.s);
    }
    CudaCheckErrorModNoSync;
}

void Plaintext::copyMetadata(const Plaintext& a) {
    assert(this->c0.getLevel() == a.c0.getLevel());
    this->slots = a.slots;
    this->NoiseLevel = a.NoiseLevel;
    this->NoiseFactor = a.NoiseFactor;
}

void Plaintext::addMetadata(const Plaintext& a, const Plaintext& b) {
    assert(this->c0.getLevel() == a.c0.getLevel());
    if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
        assert(a.c0.getLevel() == b.c0.getLevel());
        assert(a.NoiseFactor == b.NoiseFactor);
    }
    this->slots = std::max(a.slots, b.slots);
    assert(a.NoiseLevel == b.NoiseLevel);
    this->NoiseLevel = a.NoiseLevel;
    this->NoiseFactor = a.NoiseFactor;
}

void Plaintext::multMetadata(const Plaintext& a, const Plaintext& b) {
    assert(this->c0.getLevel() == a.c0.getLevel());
    if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
        assert(a.c0.getLevel() == b.c0.getLevel());
        assert(a.NoiseFactor == b.NoiseFactor);
    }
    this->slots = std::max(a.slots, b.slots);
    assert(a.NoiseLevel == b.NoiseLevel);
    this->NoiseLevel = a.NoiseLevel + b.NoiseLevel;
    this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

Plaintext::Plaintext(Context& cc)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(this->cc, -1, false, true) {
    CudaNvtxStop();
    c0.onSizeChanged = [] { CudaNvtxLifetimeRefresh(Plaintext::loc); };
    CudaNvtxLifetimeRegisterBytes(loc, this, [this] { return this->getMemoryUsage(); });
}

Plaintext::Plaintext(Context& cc, const RawPlainText& raw)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(this->cc, -1, false, true) {
    load(raw);
    CudaNvtxStop();
    c0.onSizeChanged = [] { CudaNvtxLifetimeRefresh(Plaintext::loc); };
    CudaNvtxLifetimeRegisterBytes(loc, this, [this] { return this->getMemoryUsage(); });
}

Plaintext::Plaintext(Plaintext&& pt) noexcept
: my_range(std::move(pt.my_range)), cc_(pt.cc_), cc(pt.cc), c0(std::move(pt.c0)), NoiseFactor(pt.NoiseFactor), NoiseLevel(pt.NoiseLevel),
  slots(pt.slots) {
    // Carry the NVTX lifetime registration over to the destination object.
    CudaNvtxLifetimeUnregisterBytes(loc, &pt);
    c0.onSizeChanged = [] { CudaNvtxLifetimeRefresh(Plaintext::loc); };
    CudaNvtxLifetimeRegisterBytes(loc, this, [this] { return this->getMemoryUsage(); });
}

Plaintext::~Plaintext() {
    // Drop the registry entry while the members (c0) are still alive.
    CudaNvtxLifetimeUnregisterBytes(loc, this);
}

uint64_t Plaintext::getMemoryUsage() const {
    return c0.getBytes();
}

void Plaintext::load(const RawPlainText& raw) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    NoiseFactor = raw.Noise;
    NoiseLevel = raw.NoiseLevel;
    slots = raw.slots;

    c0.loadConstant(raw.sub_0, raw.moduli, slots, true);

    /*
    cudaDeviceSynchronize();
    for (auto& i : c0.GPU) {
        for (auto& j : i.limb) {
            SWITCH(j, printThisLimb(1));
        }
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();
    */
}

void Plaintext::store(RawPlainText& raw) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    cudaDeviceSynchronize();

    raw.numRes = c0.getLevel() + 1;
    raw.sub_0.resize(raw.numRes);
    c0.store(raw.sub_0);
    raw.N = cc.N;
    c0.sync();

    raw.Noise = NoiseFactor;
    raw.NoiseLevel = NoiseLevel;
    raw.slots = slots;
    cudaDeviceSynchronize();
}

void Plaintext::moddown() {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    c0.moddown(true, true);
}

void Plaintext::raiseDegreeToLevel(int targetLevel) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    // q_l = the last tower modulus of the target level (mirrors the reference's
    // targetPoly.GetElementAtIndex(numElems - 1).GetModulus()).
    const uint32_t numTowers  = static_cast<uint32_t>(c0.getLevel()) + 1;
    const uint32_t tNumTowers = static_cast<uint32_t>(targetLevel) + 1;
    const uint64_t q          = (tNumTowers - 1 < cc.prime.size())
                                    ? cc.prime[tNumTowers - 1].p
                                    : cc.specialPrime[tNumTowers - 1 - cc.prime.size()].p;
    std::vector<uint64_t> crtQ(numTowers);
    for (uint32_t i = 0; i < numTowers; ++i) {
        const uint64_t m = (i < cc.prime.size()) ? cc.prime[i].p : cc.specialPrime[i - cc.prime.size()].p;
        crtQ[i]          = q % m;
    }
    c0.multScalar(crtQ);
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
}

bool Plaintext::adjustScaleAndLevel(const int scaleDegree, const int level, const double scaling_factor) {
    assert(scaleDegree == 2 ? cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] == cc.param.ScalingFactorRealBig[level] : true);
    //	assert(scaleDegree < 3 ? scaling_factor == (scaleDegree == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) : true);

    uint32_t c1lvl = c0.getLevel();
    uint32_t c2lvl = level;
    uint32_t c1depth = this->NoiseLevel;
    uint32_t c2depth = scaleDegree;
    auto sizeQl1 = c1lvl + 1;
    // auto sizeQl2 = c2lvl + 1;

    // OpenFHE 4a31f521 (AdjustLevelsAndDepthInPlace, first FIXED branch): under FIXEDMANUAL/FIXEDAUTO
    // operands of equal noise-scale degree already carry the same power of Delta, so aligning them is
    // only a tower drop (mirrors Ciphertext::adjustScaleAndLevel and the reference).
    if ((cc.rescaleTechnique == FIXEDMANUAL || cc.rescaleTechnique == FIXEDAUTO) && c1depth == c2depth) {
        if (c1lvl > c2lvl)
            this->dropToLevel(static_cast<int>(c2lvl), true);
        else if (c1lvl < c2lvl)
            return false;
        return true;
    }

    // OpenFHE 4a31f521 (AdjustLevelsAndDepthInPlace, FIXEDAUTO degree-mismatch branch; equal degrees
    // were handled above). A degree-2 operand is rescaled when the other has degree 1 and more towers
    // dropped; otherwise the degree-1 operand is multiplied by the target's last-tower prime q_l,
    // which the eventual rescale divides out exactly (Delta would leave a ~2^-33 residual). Mirrors
    // Ciphertext::adjustScaleAndLevel; the plaintext is the degree-1 operand in the addPt/multPt paths.
    if (cc.rescaleTechnique == FIXEDAUTO) {
        if (c1lvl == c2lvl) {
            if (c1depth < c2depth) {
                this->raiseDegreeToLevel(static_cast<int>(c2lvl));
                return true;
            }
            return false;
        }
        if (c1lvl > c2lvl) {
            if (this->NoiseLevel == 2)
                this->rescale();
            else
                this->raiseDegreeToLevel(static_cast<int>(c2lvl));
            if (c0.getLevel() > static_cast<int32_t>(c2lvl))
                this->dropToLevel(static_cast<int>(c2lvl), true);
            return true;
        }
        return false;
    }

    if (c1lvl > c2lvl) {
        if (c1depth == 2) {
            if (c2depth == 2) {
                double scf1 = NoiseFactor;
                double scf2 = scaling_factor;
                double scf = cc.param.ScalingFactorReal[c1lvl];    // cryptoParams->GetScalingFactorReal(c1lvl);
                double q1 = cc.param.ModReduceFactor[sizeQl1 - 1]; // cryptoParams->GetModReduceFactor(sizeQl1 - 1);

                multScalar(scf2 / scf1 * q1 / scf);
                rescale();
                if (c0.getLevel() > static_cast<int32_t>(c2lvl)) {
                    this->dropToLevel(c2lvl, true);
                }

                assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);
                NoiseFactor = scaling_factor;
            } else {
                if (c1lvl - 1 == c2lvl) {
                    rescale();
                } else {
                    double scf1 = NoiseFactor;
                    double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                    double scf = cc.param.ScalingFactorReal[c1lvl];         // cryptoParams->GetScalingFactorReal(c1lvl);
                    double q1 = cc.param.ModReduceFactor[sizeQl1 - 1];      // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                    multScalar(scf2 / scf1 * q1 / scf);
                    // NoiseFactor *= scf2 / scf1 * q1 / scf;
                    // NoiseFactor /= cc.param.ScalingFactorReal[this->c0.getLevel()];
                    rescale();
                    if (c0.getLevel() - 1 > static_cast<int32_t>(c2lvl)) {
                        this->dropToLevel(c2lvl + 1, true);
                        // LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
                    }
                    rescale();
                    assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);

                    // assert(NoiseFactor == scaling_factor);
                    NoiseFactor = scaling_factor;
                }
            }
        } else {
            if (c2depth == 2) {
                double scf1 = NoiseFactor;
                double scf2 = scaling_factor;
                double scf = cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
                multScalar(scf2 / scf1 / scf);
                this->dropToLevel(c2lvl, true);
                // LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
                assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
                NoiseFactor = scf2;
            } else {
                double scf1 = NoiseFactor;
                double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                double scf = cc.param.ScalingFactorReal[c1lvl];         // cryptoParams->GetScalingFactorReal(c1lvl);
                multScalar(scf2 / scf1 / scf);
                if (c1lvl - 1 > c2lvl) {
                    this->dropToLevel(c2lvl + 1, true);
                    // LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                }
                rescale();
                assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
                NoiseFactor = scaling_factor;
            }
        }
        return true;
    } else if (c1lvl < c2lvl) {
        return false;
    } else {
        if (c1depth < c2depth) {
            multScalar(1.0, false);
        } else if (c2depth < c1depth) {
            return false;
        }
        return true;
    }
}

bool Plaintext::adjustPlaintextToCiphertext(const Plaintext& p, const Ciphertext& c) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    // constexpr bool PRINT = false;
    /*
        if (cc.rescaleTechnique == FIXEDAUTO) {
            if (p.c0.getLevel() - p.NoiseLevel > c.getLevel() - c.NoiseLevel) {
                if (this != &p)
                    this->copy(p);
                if (c.NoiseLevel == 1 && NoiseLevel == 2) {
                    this->c0.dropToLevel(c.getLevel() + 1);
                    rescale();
                } else {
                    this->c0.dropToLevel(c.getLevel());
                }
                return true;
            } else if (c.NoiseLevel == 1 && p.NoiseLevel == 2) {
                if (this != &p)
                    this->copy(p);
                rescale();
                return true;
            } else if (p.NoiseLevel == 1 && c.NoiseLevel == 2) {
                return false;
            } else {
                this->copy(p);
                return true;
            }
        }
    */
    if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
        if (this != &p)
            this->copy(p);
        return adjustScaleAndLevel(c.NoiseLevel, c.getLevel(), c.NoiseFactor);
    }
    return false;
}

void Plaintext::copy(const Plaintext& p) {
    assert(slots == 0 || slots == p.slots);
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    this->c0.copy(p.c0, true, p.slots * 2);
    this->copyMetadata(p);
}

void Plaintext::multScalar(double c, bool rescale) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    /*
    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);
    */
    auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
    /*
    for (int i = 0; i < elem.size(); i++) {
        std::cout << elem[i] << " ";
    }
    std::cout << std::endl;
*/
    c0.multScalar(elem);

    if (rescale) {
        c0.rescale(slots);
    }
    // Manage metadata
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel() + rescale);
    if (rescale) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + rescale);
        NoiseLevel -= 1;
    }
}

void Plaintext::rotate_hoisted(const std::vector<int>& indexes, std::vector<Plaintext*>& results) {
    assert(indexes.size() == results.size() && "rotate_hoisted: mismatched indexes and results sizes");
    CKKS::SetCurrentContext(cc_);
    assert(2 * slots == cc.N);

    for (size_t i = 0; i < indexes.size(); ++i) {
        int index = indexes[i];
        if (index == 0) {
            results[i]->copy(*this);
        } else {

            // Copy and rotate

            results[i]->c0.grow(this->c0.getLevel());
            results[i]->c0.dropToLevel(this->c0.getLevel());
            results[i]->copyMetadata(*this);
            results[i]->c0.automorph(index, &this->c0, 2 * slots);
            // results[i]->copy(*this);
            // results[i]->automorph(index);
        }
    }
}

#if false
void Plaintext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (NoiseLevel == 2)
            this->rescale();
    }

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        // if (b.c0.getLevel() != this.getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
        if (!hasSameScalingFactor(b)) {
            Plaintext b_(cc);
            if (NoiseLevel == 2)
                this->rescale();
            if (b_.NoiseLevel == 2)
                b_.rescale();
            multPt(b_, rescale);
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}

void Plaintext::addPt(const Plaintext& c) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

    // assert(NoiseLevel == b.NoiseLevel);
    c0.add(c.c0);
}

#endif

void Plaintext::rescale() {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);

    assert(this->NoiseLevel >= 2);
    /*
    std::cout << "Rescale plaintext, level" << c0.getLevel() << std::endl;
    for (auto& i : c0.GPU) {
        std::cout << i.limb.size() << " ";
    }
    std::cout << std::endl;
*/
    c0.rescale(slots);

    // Manage metadata
    NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
    NoiseLevel -= 1;
}

void Plaintext::automorph(const int index) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    /*
    if (c0.isModUp()) {
        std::cout << "isModup plaintext automorph not implemented" << std::endl;
    }
    */
    if (index != 0) {
        auto& aux = cc.getModdownAux(0);
        aux.setLevel(c0.getLevel());
        aux.automorph(index, &c0, 2 * slots);
        c0.copy(aux, true, 2 * slots);
    }
}

void Plaintext::dropToLevel(const int level, bool skip_adjust) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    CKKS::SetCurrentContext(cc_);
    if (c0.getLevel() <= level)
        return;
    if (!skip_adjust && (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT)) {
        assert(NoiseLevel == 1 || NoiseLevel == 2);
        bool ok = adjustScaleAndLevel(
            this->NoiseLevel, level, this->NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : cc.param.ScalingFactorReal[level] * cc.param.ScalingFactorReal[level]);
        assert(ok);
    } else {
        c0.dropToLevel(level);
    }
}

void Plaintext::multPt(const Plaintext& b1, const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

    this->copy(b1);
    this->dropToLevel(std::min(b.c0.getLevel(), b1.c0.getLevel()), false);
    multPt(b, rescale);
    if (rescale && NoiseLevel > 1) {
        this->rescale();
    }
}

void Plaintext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

    constexpr bool PRINT = false;
    if (cc.rescaleTechnique == CKKS::FIXEDAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        if constexpr (PRINT)
            std::cout << "multPt: Rescale input ciphertext" << std::endl;
        if (NoiseLevel > 1)
            this->rescale();
    }
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        if constexpr (PRINT)
            std::cout << "multPt: Rescale input ciphertext" << std::endl;

        if (b.c0.getLevel() > c0.getLevel()) {
            Plaintext b_(cc_);
            b_.copy(b);
            if (b_.NoiseLevel == 2)
                b_.rescale();
            if (b_.c0.getLevel() > c0.getLevel()) {
                b_.dropToLevel(c0.getLevel(), false);
            }
            multPt(b_, rescale);
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    // op_count[OPS::MULTPT]++;

    assert(this->slots >= b.slots);
    c0.multPt(b.c0, false, b.slots);

    this->multMetadata(*this, b);

    if (rescale && NoiseLevel > 1) {
        this->rescale();
    }
}

void Plaintext::subPt(const Plaintext& c) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

    assert(NoiseLevel == c.NoiseLevel);

    assert(this->slots >= c.slots);
    c0.sub(c.c0, c.slots);
    this->addMetadata(*this, c);
}

} // namespace FIDESlib::CKKS