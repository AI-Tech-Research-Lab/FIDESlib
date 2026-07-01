//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

// Ciphertext offloading: evict GPU-resident ciphertexts to host RAM to free VRAM, then
// bring them back on demand. Useful when you are holding on to more ciphertexts than
// comfortably fit in GPU memory, or simply want to hand some VRAM back to the system
// before doing other GPU work.
//
//   ct->Offload()          - copy the ciphertext's limbs to host RAM and free its VRAM.
//   ct->IsOffloaded()      - whether the ciphertext is currently evicted.
//   ct->Reload()           - copy it back to the GPU (also happens automatically on first
//                            use of an offloaded ciphertext).
//   cc->TrimGPUMemoryPool()- return the freed VRAM to the driver/OS.
//
// Note: Offload()/Reload() are a bit-exact round trip - no decrypt, rescale or NTT is
// performed, so the ciphertext is numerically identical afterwards.
//
// Note: Offload() alone frees the limbs into FIDESlib's internal GPU memory pool so they
// can be cheaply reused by later operations; it does NOT return that memory to the OS
// (nvidia-smi will not show a drop). Call TrimGPUMemoryPool() when you actually want the
// VRAM back for something else. Reusing the memory for more FIDESlib work needs no trim.

#include <fideslib.hpp>

#include <iostream>
#include <vector>

using namespace fideslib;

int main() {

	// Step 1: Setup CryptoContext.

	uint32_t multDepth	  = 5;
	uint32_t scaleModSize = 50;
	uint32_t batchSize	  = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(multDepth);
	parameters.SetScalingModSize(scaleModSize);
	parameters.SetBatchSize(batchSize);
	parameters.SetDevices({ 0 });
	parameters.SetPlaintextAutoload(false);
	// Autoload an offloaded ciphertext back to the GPU automatically on first use, so you
	// do not have to call Reload() explicitly before operating on it.
	parameters.SetCiphertextAutoload(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;

	// Step 2: Key Generation and context load onto the GPU.

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->LoadContext(keys.publicKey);

	// Step 3: Encrypt a batch of ciphertexts we want to keep around for later, but which we
	// do not need on the GPU right now.

	std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	constexpr int kCached = 8;
	std::vector<Ciphertext<DCRTPoly>> cached;
	cached.reserve(kCached);
	for (int i = 0; i < kCached; ++i) {
		cached.push_back(cc->Encrypt(keys.publicKey, ptxt));
	}
	std::cout << "Encrypted " << kCached << " ciphertexts, all resident on the GPU." << std::endl;

	// Step 4: Offload them to host RAM to free VRAM for other work.

	for (auto& ct : cached) {
		ct->Offload();
	}
	std::cout << "Offloaded all " << kCached << " ciphertexts (IsOffloaded[0] = " << std::boolalpha << cached[0]->IsOffloaded() << ")." << std::endl;

	// Return the freed memory to the OS. Skip this if you only intend to reuse the memory
	// for more FIDESlib operations - the internal pool already recycles it for free.
	cc->TrimGPUMemoryPool();
	std::cout << "Trimmed the GPU memory pool: VRAM handed back to the system." << std::endl;

	// Step 5: Do other GPU work while the batch sits in host RAM. This fresh ciphertext uses
	// the VRAM the offloaded batch is no longer occupying.

	auto other	= cc->Encrypt(keys.publicKey, ptxt);
	auto doubled = cc->EvalAdd(other, other);
	Plaintext otherResult;
	cc->Decrypt(keys.secretKey, doubled, &otherResult);
	otherResult->SetLength(batchSize);
	std::cout.precision(8);
	std::cout << "Meanwhile, 2 * x on a fresh ciphertext = " << otherResult;

	// Step 6: Bring a cached ciphertext back and use it. With CiphertextAutoload enabled the
	// reload happens automatically on first use; Reload() would do it explicitly.
	auto sum = cc->EvalAdd(cached[0], cached[1]); // cached[0], cached[1] auto-reload here

	Plaintext result;
	cc->Decrypt(keys.secretKey, sum, &result);
	result->SetLength(batchSize);
	std::cout << "After reload, cached[0] + cached[1] = " << result;
	std::cout << "(each was x, so the result is 2 * x, recovered bit-exactly after offload)" << std::endl;

	return 0;
}
