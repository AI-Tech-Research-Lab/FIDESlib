# Plan: GPU ciphertext offload / reload (host ↔ device)

**Status:** design, not yet implemented.
**Audience:** a Claude Code session spawned in this repo (`AI-Tech-Research-Lab/FIDESlib`).
**Scope of this repo's work:** the C++/CUDA offload primitive on `FIDESlib::CKKS::Ciphertext`
plus GoogleTests. The Python/OpenFHE-wrapper exposure is a *follow-up in the PyFIDESlib
repo* and is described at the end but is **out of scope for the work done here**.

---

## 1. Goal & requirements

Let a GPU-resident ciphertext be **evicted from VRAM to host RAM** and **brought back**,
so a caller with more ciphertexts than fit in VRAM can stream them through the GPU.

Hard requirements (from the task owner):

1. **Offload must not touch the ciphertext.** No decrypt, no rescale, no level switch,
   no NTT/domain change — a bit-exact round trip of the RNS limbs.
2. **Ciphertexts move both ways.** `offload()` (device→host, free VRAM) and `reload()`
   (host→device). Auto-reloading on first use is acceptable and desirable (see §6).
3. **Unit tests** proving correctness (bit-exact round trip) and that VRAM is actually
   released while offloaded.

---

## 2. Why this is mostly a "free + restore" problem, not a "serialize" problem

The host↔device limb transfer **already exists** and is already bit-exact (it is what
`Decrypt` and cross-context moves rely on). The pieces:

- `FIDESlib::CKKS::Ciphertext` (`src/CKKS/Ciphertext.cuh`) holds two `RNSPoly c0, c1`
  plus metadata (`keyID`, `NoiseLevel`, `NoiseFactor`, `slots`).
- `RNSPoly` (`src/CKKS/RNSPoly.cuh`) holds `std::vector<LimbPartition> GPU` and has:
  - `store(std::vector<std::vector<uint64_t>>& data)` — copies each GPU limb → host
    (`RNSPoly.cpp`, uses `SWITCH(... store_convert(data[i]))`).
  - `load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli)`
    — `grow`/`dropToLevel` to the right level, then copies host → GPU limb-by-limb.
- `Ciphertext::store(RawCipherText&)` / `Ciphertext::load(const RawCipherText&)`
  (`Ciphertext.cpp:311-342`) wrap the above and also carry the metadata + moduli.
  `RawCipherText` (`src/CKKS/openfhe-interface/RawCiphertext.cuh:19`) is the host struct
  (`sub_0`, `sub_1` = `vector<vector<uint64_t>>`, `moduli`, `numRes`, `N`, `Noise`,
  `NoiseLevel`, `slots`).

**These transfers are raw RNS limbs** — no decode, no NTT, no rescale. So requirement #1
is satisfied for free *provided we do not call any leveling op between store and load*.

What is **missing** is the ability to **release the VRAM** after storing and to bring the
object back to a loadable-but-empty state. That is the new primitive.

### Memory ownership (where the VRAM actually is)
`RNSPoly.GPU[g]` is a `LimbPartition` (`src/CKKS/LimbPartition.cuh`) holding
`std::vector<LimbImpl> limb` (per-limb device buffers) and, in single-malloc mode, a
shared `uint64_t* bufferLIMB`. Underlying device memory is a `VectorGPU<T>`
(`src/VectorGPU.cuh`) whose `free(Stream&)` / destructor release it. Existing frees:
`LimbPartition::dropLimb()` (`LimbPartition.cu:2290`, pops one limb) and
`~LimbPartition()` (frees `limbptr`/`auxptr`/`bufferLIMB`/…). `RNSPoly::dropToLevel`
uses `dropLimb` in the per-limb (non-single-malloc) path.

---

## 3. Design

Add a host-side snapshot to the ciphertext and a free/restore pair on each layer.

### 3.1 Data held while offloaded
Reuse the existing serialized form. Add to `Ciphertext` (or a small owned struct):

```cpp
struct OffloadedState {                 // host-resident, no GPU handles
    RawCipherText raw;                  // sub_0, sub_1, moduli, metadata (already exists)
    bool present = false;
};
```

Storing the whole `RawCipherText` reuses `Ciphertext::store` verbatim and captures
`moduli` (needed by `RNSPoly::load`). Keep it simple first; optimize the host buffer
later (see §8 "pinned memory").

### 3.2 New methods

**`Ciphertext`** (`src/CKKS/Ciphertext.{cuh,cpp}`):
- `void offload();`
  1. If already offloaded, return.
  2. `store(state.raw);` (device→host; already syncs — see `store` calls
     `cudaDeviceSynchronize()`).
  3. Free the GPU limbs of `c0` and `c1` via the new `RNSPoly::freeGPU()` (§3.3).
  4. `state.present = true;`
- `void reload();`
  1. If not offloaded, return.
  2. `c0.load(state.raw.sub_0, state.raw.moduli);` /
     `c1.load(state.raw.sub_1, state.raw.moduli);` — `load` will `grow` from the freed
     (empty) state back to the original level and copy host→GPU.
  3. Restore metadata (already inside `Ciphertext::load`), clear `state`.
- `[[nodiscard]] bool isOffloaded() const;`

**`RNSPoly`** (`src/CKKS/RNSPoly.{cuh,cpp}`):
- `void freeGPU();` — release all limb device memory but keep the object valid to
  `load`/`grow` again. Set internal `level = -1`. Must handle **both** allocation modes:
  - per-limb (`bufferLIMB == nullptr`): drop every limb (`while (limb.size()) dropLimb();`)
    across all `GPU[g]`.
  - single-malloc (`bufferLIMB != nullptr`): free the shared buffer and clear the
    `limb` vector / pointers. **Verify** how `~LimbPartition` frees `bufferLIMB` and mirror
    exactly (do not double-free).

**`LimbPartition`** (`src/CKKS/LimbPartition.{cuh,cu}`):
- `void freeLimbs();` — the single-partition worker for the above, factoring the
  buffer-mode logic out of the destructor so both can share it.

### 3.3 The "does not touch the ciphertext" guarantee
`offload` calls only `store` (raw copy) + `freeGPU` (dealloc). `reload` calls only
`load` (raw copy + `grow`). None of rescale / modDown / modUp / NTT / dropToLevel-to-a-
*higher* level is invoked, so the logical ciphertext (level, scale, domain, value) is
identical before and after. The unit test in §5 pins this down bit-for-bit.

---

## 4. Edge cases to handle (call these out in the implementation)

- **Special/aux limbs & modUp:** a normal stored ciphertext is not in `modUp` state and
  has no special limbs (those exist transiently inside key-switch). `Ciphertext::store`
  only stores `level+1` limbs. **Assert** `!c0.isModUp() && !c1.isModUp()` at the top of
  `offload()` and fail loudly if violated rather than silently corrupting.
- **Single-malloc vs per-limb allocation** (§3.2) — the main correctness risk. Confirm the
  freeing path for `bufferLIMB` and the `*ptr` `VectorGPU`s.
- **Multi-GPU:** `RNSPoly::GPU` can hold several `LimbPartition`s; `store`/`load` already
  iterate `cc.limbGPUid` and set the device. `freeGPU` must loop all `GPU[g]` and
  `cudaSetDevice(g.device)` before freeing (mirror `~LimbPartition`).
- **Streams / sync:** `store` already `cudaDeviceSynchronize()`s before/after. Ensure
  `freeGPU` waits on the partition stream (`dropLimb` does `STREAM(limb.back()).wait(s)`)
  so we do not free memory a pending copy still reads.
- **Double offload / double reload / destruct-while-offloaded:** make idempotent; the
  destructor must not double-free freed limbs.
- **Level 0 ciphertext** (single limb): make sure `freeGPU` down to empty and `load`'s
  `grow` back to level 0 works (there is a `grow` guard `if (level >= new_level) return;`
  and a known "grow from -1" path — test this boundary).

---

## 5. Testing (C++, in this repo — REQUIRED)

Framework is GoogleTest with the parametrized fixture `GeneralParametrizedTest`
(`test/ParametrizedTest.cuh`). Model the round trip on
`test/OpenFheInterfaceTests.cu` (see `TEST_P(OpenFHEInterfaceTest, ExtractContextShowAdd)`,
lines 31-89): OpenFHE `Encrypt` → `GetRawCipherText` → `FIDESlib::CKKS::Ciphertext(cc_, raw)`
→ GPU ops → `store` → `GetOpenFHECipherText` → `Decrypt` → `ASSERT_EQ_CIPHERTEXT`.

Add `test/CiphertextOffloadTests.cu` (and register it in `test/CMakeLists.txt` next to the
other `*Tests.cu`). Tests:

1. **Bit-exact identity round trip.** Build `GPUct` from a known plaintext. Take a
   reference `store` → `RawCipherText A`. Then `GPUct.offload(); GPUct.reload();` and
   `store` → `RawCipherText B`. Assert `A.sub_0 == B.sub_0`, `A.sub_1 == B.sub_1`, and all
   metadata equal (raw `uint64_t` limb equality — stronger than a decrypt tolerance check).
2. **Correctness after reload.** offload → reload → `Decrypt`, compare to the original
   plaintext with the usual `ASSERT_EQ_CIPHERTEXT` / tolerance helper.
3. **Compute-after-reload.** offload → reload → GPU `add`/`mult` → decrypt, compare to the
   CPU (`cc->EvalAdd`/`EvalMult`) result. Proves the reloaded object is fully functional.
4. **VRAM is actually released.** Bracket with `cudaMemGetInfo(&free_before, &total)`:
   record free VRAM with ct resident, `offload()`, `cudaDeviceSynchronize()`, record again;
   assert free VRAM increased by roughly the ciphertext footprint
   (`~2 * (level+1) * N * sizeof(uint64_t)`), with a slack tolerance for allocator
   granularity/caching. (No existing `cudaMemGetInfo` use in the tree — this is new.)
5. **Idempotence / boundaries.** double `offload()`, double `reload()`, offload a level-0
   ciphertext, and offload one that has already been rescaled to a low level.

Build & run (see `VENDORED.md` / repo README for the CUDA build; adjust arch):
```bash
cd build && cmake --build . -j --target <test_binary>
ctest -R Offload --output-on-failure     # or run the gtest binary with --gtest_filter=*Offload*
```

---

## 6. Auto-reload (requirement #2, optional but wanted)

Cheapest correct version: at the **top of every mutating `Ciphertext` op** that touches
`c0`/`c1` (`add`, `sub`, `mult`, `multPt`, `rotate`, `store`, …) add
`if (state.present) reload();`. This makes "moved automatically to the GPU when invoked"
true. Prefer a single private helper `ensureResident()` called at the entry of each op,
rather than sprinkling the flag check.

Do **not** auto-*offload* — eviction is a capacity decision the caller makes explicitly.

Keep auto-reload behind the same `OffloadedState::present` flag so an offloaded ct is never
silently used with freed pointers.

---

## 7. Suggested commit breakdown (keep PR-able, like the existing fix commits)

1. `RNSPoly::freeGPU` + `LimbPartition::freeLimbs` (factor buffer-free out of destructor).
2. `Ciphertext::offload` / `reload` / `isOffloaded` + `OffloadedState`.
3. `ensureResident()` auto-reload hooks in the mutating ops.
4. `test/CiphertextOffloadTests.cu` + CMake registration.

Each commit message ends with the repo trailer:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 8. Follow-ups / non-goals for this repo

- **Python / OpenFHE wrapper exposure (separate repo: `AI-Tech-Research-Lab/PyFIDESlib`).**
  The wrapper drives OpenFHE's `CryptoContext` + `CiphertextImpl<DCRTPoly>` and already
  exposes `LoadCiphertext`, `SetAutoLoadCiphertexts`, `SetCiphertextAutoload` (these live in
  the **patched OpenFHE**, not in this repo). The GPU-resident FIDESlib ciphertext that
  backs an OpenFHE ct is managed there. Exposing `offload()` at the Python level means
  wiring it into that CryptoContext/registry layer and adding a `.def("Offload", …)` in
  `PyFIDESlib/src/bindings.cpp` (Ciphertext class, `bindings.cpp:140`). Do that after the
  C++ primitive lands and is tested here.
- **Pinned host memory** (`cudaMallocHost`) for the offload buffer to make transfers async
  and faster — optimization, not needed for correctness. No pinned-alloc usage exists in the
  tree yet.
- **Overlap / prefetch** (offload N+1 while computing on N) — future.

---

## 9. First steps for the spawned agent (verify before coding)

1. Read `RNSPoly::store`/`load`, `Ciphertext::store`/`load`, `RNSPoly::grow`/`dropToLevel`,
   `LimbPartition::dropLimb` + `~LimbPartition`, and `VectorGPU::free`/dtor. Confirm the
   single-malloc (`bufferLIMB`) free path precisely — this is the one real unknown.
2. Confirm a stored, non-key-switching ciphertext never carries special limbs / `modUp`.
3. Confirm `RNSPoly::load` correctly regrows from a fully-freed (`level == -1`) state at
   level 0 and at a high level (the `grow` "from -1" path).
4. Only then implement §3 and §5.
