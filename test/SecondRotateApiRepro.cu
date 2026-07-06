//
// Fast C++ repro (via the api:: OpenFHE-style layer, not the raw FIDESlib::CKKS:: core) for the
// "second EvalRotate chained through an intervening EvalMult NaNs on first-ever use" bug from
// fideslib_second_rotate_bug_mwe.py. Confirmed still present on top of commit 3df8889 (the
// first-key-switch-NaN fix).
//
// Unlike a direct FIDESlib::CKKS::Ciphertext test (which reuses one persistent C++ object and
// never reproduces this), the api:: layer's EvalRotate/EvalMult each deep-copy into a FRESH
// FIDESlib::CKKS::Ciphertext (CryptoContextImpl::CopyDeviceCiphertext, api/CryptoContext.cpp) and
// evict+destroy the previous one the moment its shared_ptr refcount drops (CiphertextImpl::~dtor
// -> EvictDeviceCiphertext -> map erase -> FIDESlib::CKKS::Ciphertext::~Ciphertext ->
// cc.returnAuxilarPoly(...), src/CKKS/Ciphertext.cpp). That destroy/recycle path
// (ContextData::getAuxilarPoly/returnAuxilarPoly, src/CKKS/Context.cu) hands the raw RNSPoly
// buffer straight to the next ciphertext construction with **no synchronization at all** --
// exactly the kind of cross-stream-reuse-while-writes-in-flight hazard the morning's fix closed
// for the key-switch scratch, but here for ordinary per-op ciphertext buffers.
//
#include <openfhe.h>
#undef duration
#include <fideslib.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace fideslib;

namespace {

bool run_trial(int sink_levels, std::vector<double>& head, double& max_abs_err) {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetSecurityLevel(HEStd_NotSet);
	params.SetRingDim(1 << 16);
	params.SetMultiplicativeDepth(30);
	params.SetScalingModSize(59);
	params.SetFirstModSize(60);
	params.SetNumLargeDigits(3);
	params.SetBatchSize(1024);
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetKeySwitchTechnique(HYBRID);
	params.SetSecretKeyDist(UNIFORM_TERNARY);
	params.SetDevices({ 0 });

	CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });
	cc->LoadContext(keys.publicKey);

	std::vector<double> x(1024);
	for (int i = 0; i < 1024; ++i)
		x[i] = (i < 8) ? (i + 1) : 0.0;
	Plaintext pt					= cc->MakeCKKSPackedPlaintext(x);
	Ciphertext<DCRTPoly> ct			= cc->Encrypt(keys.publicKey, pt);
	Plaintext one_pt				= cc->MakeCKKSPackedPlaintext(std::vector<double>(1024, 1.0));

	for (int i = 0; i < sink_levels; ++i)
		ct = cc->EvalMult(ct, one_pt);

	Ciphertext<DCRTPoly> r1  = cc->EvalRotate(ct, 1);
	Ciphertext<DCRTPoly> mid = cc->EvalMult(r1, one_pt);
	Ciphertext<DCRTPoly> r2  = cc->EvalRotate(mid, 1);

	Plaintext result;
	cc->Decrypt(keys.secretKey, r2, &result);
	result->SetLength(1024);

	bool has_nan = false;
	head.clear();
	max_abs_err = 0.0;
	for (int i = 0; i < 1024; ++i) {
		double v = result->GetRealPackedValue().at(i);
		if (i < 8)
			head.push_back(v);
		if (std::isnan(v))
			has_nan = true;
		else {
			// rotate-left-by-2 of x: slot i should hold x[(i + 2) % 1024]
			int src	   = (i + 2) % 1024;
			double exp = (src < 8) ? (src + 1) : 0.0;
			max_abs_err = std::max(max_abs_err, std::abs(v - exp));
		}
	}
	return has_nan;
}

} // namespace

TEST(SecondRotateApiRepro, SinkZeroClean) {
	std::vector<double> head;
	double err;
	bool nan = run_trial(0, head, err);
	std::cout << "sink_levels=0 nan=" << nan << " max_abs_err=" << err << std::endl;
	EXPECT_FALSE(nan);
}

TEST(SecondRotateApiRepro, SinkThreeReproducesNaN) {
	std::vector<double> head;
	double err;
	bool nan = run_trial(3, head, err);
	std::cout << "sink_levels=3 nan=" << nan << " max_abs_err=" << err << " head=[";
	for (double v : head)
		std::cout << v << " ";
	std::cout << "]" << std::endl;
	EXPECT_FALSE(nan) << "second rotate (via api:: layer) returned NaN at sink_levels=3";
}
