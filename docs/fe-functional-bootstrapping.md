# FE functional bootstrapping

Fourier-extension functional bootstrapping (FE-FBT) refreshes a CKKS ciphertext and, in the
same pass, evaluates a smooth real function `f` on its slots (sigmoid, exp, GELU, ...) to
about 30 bits of precision. It is OpenFHE 1.6.1's `EvalFEFuncBootstrap*` (ePrint 2026/367),
brought to the GPU: every GPU result is **bit-identical** to the OpenFHE CPU reference, under
FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO and FIXEDMANUAL and with UNIFORM_TERNARY,
SPARSE_TERNARY and SPARSE_ENCAPSULATED secret keys.

Read the [security notes](#security-notes) before using it on real data.

## API

| Call | Role |
|---|---|
| `cc->EvalFEFuncBootstrapSetup(levelBudget, dim1, slots)` | CoeffsToSlots/SlotsToCoeffs precomputation; before `LoadContext()` |
| `cc->EvalBootstrapKeyGen(sk, slots)` | the rotation/conjugation (and sparse-switch) keys, as for any bootstrap |
| `cc->EvalFEFuncBootstrap(ct, coefficients)` | refresh + one function |
| `cc->EvalFEFuncBootstrapPrecompute(ct, coefficients)` | the function-independent part, returns a `SeriesPowers` handle |
| `cc->EvalFEFuncBootstrapWithPrecomp(powers, coefficients)` | one function on shared powers |
| `FHECKKSRNS::GetFEFBTDepth(levelBudget, coefficients, skd, firstModSize)` | depth to size the context with |

Names and semantics are OpenFHE's. Every call works both before `LoadContext()` (on the CPU,
through OpenFHE) and after it (on the GPU); the input ciphertext is never modified.

```cpp
#include <fideslib.hpp>
using namespace fideslib;

const std::vector<uint32_t> levelBudget = { 3, 2 };
const SecretKeyDist skd = SPARSE_ENCAPSULATED;
const uint32_t depth = FHECKKSRNS::GetFEFBTDepth(levelBudget, sigmoidCoeffs, skd, 60) + levelsAfter;

CCParams<CryptoContextCKKSRNS> p;
p.SetSecretKeyDist(skd);
p.SetScalingModSize(59);
p.SetFirstModSize(60);            // required: FirstModSize = ScalingModSize + 1
p.SetMultiplicativeDepth(depth);
p.SetBatchSize(slots);
// ... ring dimension / security level, HYBRID key switching (the default)
auto cc = GenCryptoContext(p);
for (auto f : { PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE })
    cc->Enable(f);

cc->EvalFEFuncBootstrapSetup(levelBudget, { 0, 0 }, slots);
auto keys = cc->KeyGen();
cc->EvalMultKeyGen(keys.secretKey);
cc->EvalBootstrapKeyGen(keys.secretKey, slots);
cc->SetDevices({ 0 });
cc->LoadContext(keys.publicKey);

// Message in [-1/2, 1/2), at the depletion level (see "Input level").
auto pt = cc->MakeCKKSPackedPlaintext(m, 1, depth - (levelBudget[1] + 1), nullptr, slots);
auto ct = cc->Encrypt(keys.publicKey, pt);

auto y = cc->EvalFEFuncBootstrap(ct, sigmoidCoeffs);          // one function

auto powers = cc->EvalFEFuncBootstrapPrecompute(ct, geluCoeffs); // several functions,
auto a = cc->EvalFEFuncBootstrapWithPrecomp(powers, expCoeffs);  // one refresh
auto b = cc->EvalFEFuncBootstrapWithPrecomp(powers, sigmoidCoeffs);
```

`examples/fe-functional-bootstrap` is a complete program (one precomputation, three functions,
precision and timings).

## Coefficients

The series is a Fourier extension of `f` over its domain, not a Chebyshev series. A function `f`
on `[-B, B]` is evaluated on a message `m` in `[-1/2, 1/2)` as `g(m) = f(2 B m)`; the bootstrap
computes

    2 * Re( c_0 + sum_{j >= 1} c_j * exp(2 * pi * i * j * t) ),   t = m / 2,

so `coefficients` is `c_0, c_1, ..., c_d` (complex). The coefficients of the OpenFHE example
(exp on `[-2, 2]`, degree 29; sigmoid and GELU on `[-8, 8]`, degrees 34 and 44) are in
`examples/fe-functional-bootstrap` and `test/FeFuncCoefficients.hpp`. To generate others, use the
scripts of [openfheorg/fefbt-python](https://github.com/openfheorg/fefbt-python); FIDESlib does not
ship a generator.

The output is always real. With `CKKSDataType::COMPLEX` the imaginary part of every input slot is
discarded, and the output slots have zero imaginary part.

## Sizing the depth

`FHECKKSRNS::GetFEFBTDepth(levelBudget, coefficients, skd, firstModSize)` returns the depth the
bootstrap consumes: both linear transforms, the complex exponential (whose Chebyshev table and
number of double-angle squarings depend on the secret distribution and, for SPARSE_ENCAPSULATED,
on `firstModSize`) and the series. Set the context depth to it plus the levels you want left on
the refreshed ciphertext; for a family of functions sharing one precomputation, use the longest
series. With `depth = GetFEFBTDepth(...) + k` the result keeps `k + levelBudget[1]` levels (the
SlotsToCoeffs levels are spent before the modulus raise), one fewer for a series below degree 5.

What the GPU checks before it touches the ciphertext is exact: the levels left after the modulus
raise must cover

    levelBudget[0] + depth(exponential table) + double-angle squarings + depth(series)

where `depth(series)` is OpenFHE's normalized depth for a Paterson-Stockmeyer series (degree 5 or
more) and one level more than it for a shorter series (OpenFHE's `GetFEFBTDepth` undercounts the
linear path by one level; its `levelBudget[1]` margin hides it). At that bound the result is a
ciphertext at its last decryptable level, which only fits function values of magnitude below
about 1 (for example sigmoid); one level below it the GPU throws `std::invalid_argument`. Size
with `GetFEFBTDepth` and a margin, not with the bound.

## Input level

The input is first brought down to the level the SlotsToCoeffs matrices were encoded for,
`L0 - (levelBudget[1] + 2)` towers dropped (one more under FLEXIBLEAUTOEXT); encoding it with
`MakeCKKSPackedPlaintext(m, 1, depth - (levelBudget[1] + 1), ...)` puts it exactly there. A
fresher input is depleted to that level (with the scale correction of FLEXIBLEAUTO); a deeper
one is rejected with "Not enough levels to perform FE functional bootstrapping".

## Requirements and errors

- HYBRID key switching and `FirstModSize == ScalingModSize + 1` (OpenFHE only discovers the
  latter at the first evaluation; FIDESlib rejects it in the setup).
- `EvalFEFuncBootstrapSetup` must run before `LoadContext()`, with exactly two entries in
  `levelBudget` and `dim1`, and `slots` either 0 (full packing) or a power of two up to `N/2`.
- A slot count holds one bootstrapping precomputation: one set up for FE functional
  bootstrapping cannot be used by `EvalBootstrap` and vice versa (the two fold a different
  overflow bound into the CoeffsToSlots matrix), and a conflicting second setup throws.
- Coefficients must be finite, with a nonzero coefficient above the constant term. A series below
  degree 5 must not end in zeros (OpenFHE crashes on it; FIDESlib rejects it on both backends);
  longer series are evaluated trimmed.
- `EvalFEFuncBootstrapPrecompute` needs a series of degree 5 or more; the powers then serve any
  series of degree below `k * (2^m - 1)` for its Paterson-Stockmeyer shape `(k, m)`, and any
  shorter series of degree up to `k`.
- Powers built on the CPU (before `LoadContext()`) cannot be used after it, nor the other way
  round, and powers of another context are rejected.

FIDESlib checks all of this up front, before any work is done, on both backends where it is a
property of the arguments; the level checks are done by the GPU core. Errors are exceptions,
never undefined behavior on the GPU.

## Memory

The bootstrap matrices and rotation keys are the same as for any bootstrap, so
`SetBootstrapCache()` and `SetRotationKeyCache()` bound them as usual. A `SeriesPowers` handle
built on the GPU holds the powers of the complex exponential in VRAM until it is destroyed. The
powers do not go through the ciphertext registry: they are not counted in the
`SetCiphertextCache()` budget and cannot be offloaded to host RAM (the exponential the handle
keeps as its template is an ordinary ciphertext and can). The handle keeps its context alive, so
destroying the context first is safe; the GPU memory is released when the last handle goes.

## Limitations

- The series is evaluated with the serial, bit-exact (non-fused) Paterson-Stockmeyer
  transcription. The fused GPU evaluator is not used: it does not match the reference on this
  pipeline's inputs.
- Not supported: composite scaling, an iterative variant, the ModRaise-first variant and
  `budgetDec` of the upstream OpenFHE port branch, and functional bootstrapping of look-up
  tables (`EvalFBT`).

## Security notes

These are not verifiable under encryption; they are the caller's responsibility.

- **Message range.** The message must lie in `[-1/2, 1/2)`. Outside it the result is silently
  wrong; nothing throws.
- **Secret distribution.** `SPARSE_ENCAPSULATED` is the recommended distribution (probability of
  a mod-raise overflow failure below 2^-128). `UNIFORM_TERNARY` fails with probability about
  2^-79 at N = 2^16 and 2^-33 at N = 2^17; `SPARSE_TERNARY` (about 2^-23) is discouraged.
- **Decrypted outputs.** If an output is decrypted and the plaintext is shared, CKKS needs noise
  flooding to be IND-CPA^D secure; this is not done for you.
- **Parameters.** The parameters of the OpenFHE example (N = 2^16, `HEStd_NotSet`, about 27 levels
  of 59 bits) do **not** reach 128-bit security. In production, set a security level and let
  OpenFHE choose the ring dimension (N = 2^17 at these depths).

## Verification

`test/FeFuncBootstrapTests.cu` compares every GPU result with the OpenFHE CPU reference with
`ASSERT_EQ_CIPHERTEXT` (noise degree, towers and every limb), decrypts both and checks them
against `f` in the clear (error below 1e-7; about 3e-10 in practice at N = 2^12):

- `FeFuncPrimitives.*`: the complex Chebyshev series on the three exponential tables and the
  power-series evaluation (shared powers, linear series) in isolation, under all four techniques.
- `FeFuncBootstrap.*`: sparse and full packing, level budgets `{3,2}` and `{1,1}`, the three
  distributions, the four techniques, exp/sigmoid/GELU, inputs above the depletion level,
  `CKKSDataType::COMPLEX`, shared powers (three functions, twice each), short and padded
  series, the level boundary (minimum depth bit-exact, one below rejected) and the error paths.
- `ForkCacheIntegration.FeFunc*`: zero bootstrap/rotation-key/ciphertext budgets with offloaded
  inputs and exponentials, and power handles that outlive their context's scope
  (SPARSE_ENCAPSULATED, auxiliary sparse context released).

All of them also pass on two GPUs (`--devices=0,1`).
