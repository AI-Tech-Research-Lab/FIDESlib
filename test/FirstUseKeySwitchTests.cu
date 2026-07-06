//
// Regression test for FIRST_USE_KEYSWITCH_NAN_BUG.md.
//
// Before the fix, the process's first key-switch (here a single EvalRotate) on a
// ciphertext at level >= 2 returned all-NaN: the key-switch scratch is allocated
// once, lazily, from the caching GPU pool, recycling chunks whose previous tenant
// (the preceding rescale scratch) still had writes in flight -> corruption. The
// identical op run a second time (warm scratch, no pool alloc) was correct.
//
// The fix is a one-time cudaDeviceSynchronize() in the first-creation branch of
// getKeySwitchAux/getKeySwitchAux2/getModdownAux (src/CKKS/Context.cu).
//
// This test uses MWE-exact params (depth 30) so it always gets a fresh GPU
// context, i.e. it reproduces regardless of test ordering. It fails without the
// fix and passes with it. Level is overridable via env REPRO_LEVEL (default 3).
//

#include <openfhe.h>
#undef duration
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>

namespace FIDESlib::Testing {
class FirstUseKeySwitchTest : public GeneralParametrizedTest {};

static bool gpu_rotate_has_nan(FIDESlib::CKKS::Context& cc_, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int level, int batchSize, std::vector<double>& head) {
	// x = 1..8, sink to `level` with plaintext mults ON THE GPU (NOT key-switches),
	// exactly like the MWE (cc.EvalMult(ct, one)).
	std::vector<double> x(8);
	for (int i = 0; i < 8; ++i)
		x[i] = i + 1;
	lbcrypto::Plaintext ones_pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(8, 1.0));
	auto c1						= cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(x));

	FIDESlib::CKKS::RawCipherText raw1	 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw_ones = FIDESlib::CKKS::GetRawPlainText(cc, ones_pt);
	FIDESlib::CKKS::Ciphertext GPUct(cc_, raw1);
	FIDESlib::CKKS::Plaintext GPUones(cc_, raw_ones);

	for (int i = 0; i < level; ++i)
		GPUct.multPt(GPUones, true); // plaintext mult + rescale -> drops one level

	std::cout << "REPRO gpu level before rotate = " << GPUct.getLevel() << std::endl;
	GPUct.rotate(1, true); // <-- the process's first key-switch

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct.store(raw_res);
	auto cResGPU = c1->Clone();
	GetOpenFHECipherText(cResGPU, raw_res);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	resultGPU->SetLength(batchSize);

	head.clear();
	bool has_nan = false;
	for (int i = 0; i < batchSize; ++i) {
		double v = resultGPU->GetRealPackedValue().at(i);
		if (i < 8)
			head.push_back(v);
		if (std::isnan(v))
			has_nan = true;
	}
	return has_nan;
}

TEST_P(FirstUseKeySwitchTest, FirstUseRotateNoNaN) {
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context& cc_		= GPUcc;
	cc_									= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
	kskEval.Initialize(rawKskEval);
	cc_->AddRotationKey(1, std::move(kskEval));

	const char* env = getenv("REPRO_LEVEL");
	int level		= env ? atoi(env) : 3;
	int bs			= generalTestParams.batchSize;

	std::vector<double> head1, head2;
	bool nan1 = gpu_rotate_has_nan(cc_, cc, keys, level, bs, head1);
	std::cout << "REPRO level=" << level << " first use : nan=" << (nan1 ? "true" : "false") << "  [";
	for (double v : head1)
		std::cout << v << " ";
	std::cout << "]" << std::endl;

	bool nan2 = gpu_rotate_has_nan(cc_, cc, keys, level, bs, head2);
	std::cout << "REPRO level=" << level << " repeat    : nan=" << (nan2 ? "true" : "false") << "  [";
	for (double v : head2)
		std::cout << v << " ";
	std::cout << "]" << std::endl;

	EXPECT_FALSE(nan1) << "first-use key-switch returned NaN at level " << level;
	EXPECT_FALSE(nan2) << "repeat key-switch returned NaN at level " << level;

	// rotate-left-by-1 of [1..8]: slots 0..6 must be 2..8 (guards against
	// finite-but-wrong output, not just NaN). Only meaningful if no NaN.
	if (!nan1)
		for (int i = 0; i < 7 && i + 1 < (int)head1.size(); ++i)
			EXPECT_NEAR(head1[i], i + 2, 1e-2) << "first-use rotate wrong at slot " << i << ", level " << level;
}

// Exact MWE params from FIRST_USE_KEYSWITCH_NAN_BUG.md: depth 30, ring 2^17,
// scaleMod 59, firstMod 60, dnum 3, 1024 slots, FLEXIBLEAUTO.
inline GeneralTestParams gparamsMWE{
	.multDepth = 30, .firstModSize = 60, .scaleModSize = 59, .batchSize = 1024, .ringDim = 1u << 17, .dnum = 3, .GPUs = devices
};
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparamsMWE{
	std::tuple(gparamsMWE, params64_17), lbcrypto::ScalingTechnique::FLEXIBLEAUTO
};

INSTANTIATE_TEST_SUITE_P(FirstUseKeySwitchTests, FirstUseKeySwitchTest, testing::Values(tparamsMWE));
} // namespace FIDESlib::Testing
