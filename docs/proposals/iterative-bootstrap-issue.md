# Add iterative (Meta-BTS) bootstrapping to the GPU path

## Summary

`EvalBootstrap(ct, numIterations, precision)` currently ignores `numIterations`
on the GPU: it always performs a single bootstrap, even when the caller asks for
2. This silently gives worse precision than the equivalent OpenFHE CPU call.
This proposes honoring `numIterations == 2` on GPU by implementing the
two-iteration Meta-BTS scheme, matching OpenFHE's CPU behavior.

## Motivation

Meta-BTS trades one extra bootstrap for a large precision gain: bootstrap the
input, compute the residual error, bootstrap that, and subtract it back out.
OpenFHE exposes this through the `numIterations` argument to `EvalBootstrap`.
FIDESlib's GPU `EvalBootstrap` accepts the same signature but drops the
argument, so GPU callers can't get the improved precision and — worse — get a
different result than the identical CPU call for the same API.

## Proposed change

Add `FIDESlib::CKKS::IterativeBootstrap(ctxt, slots, numIterations, precision, prescaled)`
and route `EvalBootstrap` / `EvalBootstrapInPlace` through it.

The two-iteration scheme mirrors OpenFHE:

1. Inner bootstrap on a copy of the input.
2. Exact power-of-two scale-up via `multIntScalar(pow2)` (no noise-factor change).
3. Scale the original input up by the same `pow2`.
4. Error = scaled inner-result − scaled input (level/scale alignment handled by
   `Ciphertext::sub`).
5. Bootstrap the error.
6. Subtract the error bootstrap from the inner bootstrap, scale back down by
   `1/pow2`.

Includes the `FIXEDMANUAL` / `FIXEDAUTO` rescale-technique forks and the
early-out when the input already carries enough limbs that a bootstrap adds
nothing.

- `numIterations <= 1` falls through to the existing single `Bootstrap`.
- `numIterations > 2` and `prescaled` combined with iteration are rejected
  (matches the scope OpenFHE actually supports here).

## Code changes

### `src/CKKS/Bootstrap.cuh` — declaration

```cpp
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void IterativeBootstrap(Ciphertext& ctxt, const int slots, const uint32_t numIterations,
  const uint32_t precision, const bool prescaled = false);
```

### `src/CKKS/Bootstrap.cu` — implementation

Add after `FIDESlib::CKKS::Bootstrap`:

```cpp
void FIDESlib::CKKS::IterativeBootstrap(Ciphertext& ctxt, const int slots, const uint32_t numIterations,
  const uint32_t precision, const bool prescaled) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });

	if (numIterations <= 1) {
		Bootstrap(ctxt, slots, prescaled);
		return;
	}
	if (numIterations != 2)
		throw std::invalid_argument("CKKS bootstrapping only supported for 1 or 2 iterations.");
	if (prescaled)
		throw std::invalid_argument("prescaled is not supported with numIterations > 1.");

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;
	const uint64_t pow2			 = uint64_t{ 1 } << precision;
	const int initLevel			 = ctxt.getLevel(); // before anything mutates ctxt

	// Inner bootstrap on a copy -- ctxt must stay intact for the early-out and for ctScaledUp.
	Ciphertext ctInit(cc_);
	ctInit.copy(ctxt);
	Bootstrap(ctInit, slots, false);
	if (ctInit.NoiseLevel == 2) // OpenFHE: ModReduceInternalInPlace(_, 1)
		ctInit.rescale();
	multIntScalar(ctInit, pow2); // exact: NoiseFactor/NoiseLevel untouched

	// Early-out: input already has at least as many limbs as a bootstrap yields.
	if (ctInit.getLevel() <= initLevel)
		return; // leave ctxt unchanged

	Ciphertext ctScaledUp(cc_);
	ctScaledUp.copy(ctxt);
	if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
		while (ctScaledUp.NoiseLevel > 1) // OpenFHE: ModReduce by (noiseDeg - 1)
			ctScaledUp.rescale();
	multIntScalar(ctScaledUp, pow2);

	Ciphertext ctDown(cc_);
	ctDown.copy(ctInit);
	if (cc.rescaleTechnique == CKKS::FIXEDAUTO)
		ctDown.dropToLevel(ctScaledUp.getLevel(), /*skip_adjust=*/true); // raw limb drop, no scale adjust

	// Error = ctDown - ctScaledUp. Ciphertext::sub (Ciphertext.cpp:205) already performs
	// the technique-correct level/scale alignment (adjustForAddOrSub for the AUTO/FLEX
	// modes, raw dropToLevel for FIXEDMANUAL -- the "implicit mod down").
	ctDown.sub(ctScaledUp);

	// Bootstrap the error, then subtract it from the initial bootstrap.
	Bootstrap(ctDown, slots, false);
	if (ctDown.NoiseLevel == 2)
		ctDown.rescale();

	ctInit.sub(ctDown);
	ctInit.multScalar(1.0 / (double)pow2); // ordinary real-scalar mult; default
											// rescale=false matches OpenFHE (result
											// is NoiseScaleDeg 2 under FIXEDMANUAL)
	ctxt.copy(ctInit);
}
```

### `api/CryptoContext.cpp` — route the API through it

In both `EvalBootstrap` and `EvalBootstrapInPlace`, replace the single-shot call:

```cpp
-	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
+	FIDESlib::CKKS::IterativeBootstrap(*res_gpu, res_gpu->slots, numIterations, precision, prescaled);
```

(The `numIterations` / `precision` arguments already exist in the OpenFHE-style
`EvalBootstrap` signature — they were previously dropped.)

Every helper called above already exists in upstream (verified at merge-base
`b097e03`): `multIntScalar` (`ApproxModEval.cuh`), `dropToLevel(int, bool skip_adjust=false)`
and `multScalar(double, bool rescale=false)` (`Ciphertext.cuh`), plus `copy` /
`sub` / `rescale` / `getLevel` / `NoiseLevel`. No additional primitives need to
be added — these three blocks are the complete diff.

## Validation

New `OpenFHEBootstrapTest.IterativeBootstrap` (parametrized over the rescale
techniques) asserts, per batch config:

- **(a)** iterative max error vs. the true message is strictly lower than
  single-shot GPU bootstrap, and
- **(b)** GPU iterative matches OpenFHE CPU iterative within tolerance
  (`2^(-logPrecision + 8)` — looser than the single-bootstrap tolerance because
  chaining two bootstraps compounds CPU/GPU rounding differences).

Add to `test/OpenFheInterfaceTests.cu`, alongside the existing
`OpenFHEBootstrapTest` cases (reuses that fixture, so it inherits the same
parametrization over rescale techniques):

```cpp
TEST_P(OpenFHEBootstrapTest, IterativeBootstrap) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->EvalMultKeyGen(keys.secretKey);

	int slots			= 1 << 4;
	uint32_t precision	= 18;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 2, 2 }, { 2, 2 }, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, raw_param.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	// CPU reference: double (Meta-BTS) iteration.
	auto cAdd = cc->EvalBootstrap(c1, 2, precision);

	lbcrypto::Plaintext result;
	std::cout << cAdd->GetLevel() << "\n";
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Result " << result;

	FIDESlib::CKKS::Context& cc_	   = GPUcc;
	cc_								   = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	///////////////////////////////////////////////////////////

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct_o(cc_, raw1);

	CudaCheckErrorMod;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;

		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		// Single-shot GPU bootstrap: baseline error to beat.
		FIDESlib::CKKS::Ciphertext GPUctSingle(cc_);
		GPUctSingle.copy(GPUct_o);
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Bootstrap(GPUctSingle, slots, false);

		FIDESlib::CKKS::RawCipherText raw_res_single;
		GPUctSingle.store(raw_res_single);
		auto cResSingle(c2);
		GetOpenFHECipherText(cResSingle, raw_res_single);
		lbcrypto::Plaintext resultSingle;
		cc->Decrypt(keys.secretKey, cResSingle, &resultSingle);

		// Iterative (2-iteration) GPU bootstrap.
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct_o);
		cudaDeviceSynchronize();
		FIDESlib::CKKS::IterativeBootstrap(GPUct1, slots, 2, precision, false);

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c2);
		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU single-shot " << resultSingle;
		std::cout << "Result GPU iterative " << resultGPU;

		CudaCheckErrorMod;

		// (a) iterative error vs. the true message is strictly lower than single-shot.
		double errSingle = 0.0, errIter = 0.0;
		for (size_t i = 0; i < x1.size(); ++i) {
			errSingle = std::max(errSingle, std::abs(resultSingle->GetRealPackedValue().at(i) - x1.at(i)));
			errIter	  = std::max(errIter, std::abs(resultGPU->GetRealPackedValue().at(i) - x1.at(i)));
		}
		std::cout << "Single-shot max error: " << errSingle << ", iterative max error: " << errIter << std::endl;
		ASSERT_LT(errIter, errSingle);

		// (b) GPU iterative matches CPU iterative within tolerance. Looser than
		// ASSERT_ERROR_OK's single-bootstrap tolerance: this test chains two bootstraps
		// (initial + error), so CPU/GPU rounding differences compound, and observed margins
		// vary run-to-run with the fresh randomness in key/noise generation.
		double cpuVsGpu = 0.0;
		for (size_t i = 0; i < result->GetSlots(); ++i)
			cpuVsGpu = std::max(cpuVsGpu, std::abs(resultGPU->GetRealPackedValue().at(i) - result->GetRealPackedValue().at(i)));
		double tolerance = pow(2.0, -result->GetLogPrecision() + 8);
		std::cout << "CPU vs GPU max error: " << cpuVsGpu << " (tolerance: " << tolerance << ")" << std::endl;
		ASSERT_LE(cpuVsGpu, tolerance);
	}
}
```

## Scope / limitations

- Only 1 and 2 iterations are supported (as in OpenFHE's common path).
- `prescaled` is not supported together with `numIterations > 1`.
