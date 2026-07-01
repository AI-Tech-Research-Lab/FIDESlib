//
// Created by carlosad on 25/03/24.
//

#include "CudaUtils.cuh"
#include <cassert>
#include <list>
#include <string>

#include "nvtx3/nvtx3.hpp"
#include <iostream>

// #include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

#include <cuda_runtime.h>

namespace FIDESlib {

extern std::vector<cudaDeviceProp> GPUprop;

struct my_domain {
	static constexpr char const* name{ "FIDESlib" };
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

std::map<std::string, std::pair<std::unique_ptr<nvtx3::unique_range_in<my_domain>>, int>> lifetimes_map;

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {

	if (cat == FUNCTION) {
		using namespace nvtx3;
		int size = msg.size();
		const event_attributes attr{ msg,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ val },
		                             category{ static_cast<unsigned int>(cat) } };

		nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	} else if (cat == LIFETIME) {

		using namespace nvtx3;
		int size      = msg.size();
		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i + 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i + 1 },
		                             category{ static_cast<unsigned int>(cat) } };
		i = i + 1;
		if (!r) {
			r = std::make_unique<unique_range_in<my_domain>>(attr);
		} else {
			*r = unique_range_in<my_domain>(attr);
		}
	}
	// nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
	if (cat == FUNCTION) {
		nvtxDomainRangePop(D);
	} else if (cat == LIFETIME) {
		using namespace nvtx3;
		int size = msg.size();

		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i - 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i - 1 },
		                             category{ static_cast<unsigned int>(cat) } };

		i = i - 1;
		if (i <= 0) {
			if (r) {
				r.reset();
			}
		} else {
			*r = unique_range_in<my_domain>(attr);
		}

		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	}
}

int getNumDevices() {
	int d;
	cudaGetDeviceCount(&d);
	return d;
};

void CudaHostSync() {
	cudaDeviceSynchronize();
}

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
	cudaGraph_t graph;
	if constexpr (capture) {
		cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
		CudaCheckErrorModNoSync;
	}
	run();
	if constexpr (capture) {
		cudaStreamEndCapture(s.ptr(), &graph);
		if (!exec) {
			cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
			// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
			CudaCheckErrorModNoSync;
		} else {
			if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
				CudaCheckErrorModNoSync;
				// only instantiate a new graph if update fails
				cudaGraphExecDestroy(exec);
				cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
				// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
				CudaCheckErrorModNoSync;
			}
		}
		cudaGraphDestroy(graph);
		cudaGraphLaunch(exec, s.ptr());
	}
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
	void Stream::wait(const Event &ev) const {
		cudaStreamWaitEvent(ptr, ev.ptr());
	}
*/
void Stream::capture_begin() {
	CudaCheckErrorMod;
	std::cout << "Hello capture" << std::endl;
	cudaStreamCaptureStatus cap;
	cudaStreamIsCapturing(ptr(), &cap);

	CudaCheckErrorMod;
	if (cap == cudaStreamCaptureStatusNone) {
		std::cout << "None" << std::endl;
		cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
	} else if (cap == cudaStreamCaptureStatusActive) {
		std::cout << "Fail: activo" << std::endl;
	} else if (cap == cudaStreamCaptureStatusInvalidated) {
		std::cout << "Fail: invalidado" << std::endl;
	} else {
		std::cout << "Fail" << std::endl;
	}
	CudaCheckErrorMod;
}

void Stream::capture_end() {
	cudaGraph_t graph;
	cudaStreamEndCapture(ptr(), &graph);
	CudaCheckErrorMod;
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
	CudaCheckErrorMod;
	cudaGraphDestroy(graph);
	CudaCheckErrorMod;
	cudaGraphLaunch(graphExec, 0);
	cudaGraphExecDestroy(graphExec);

	cudaStreamSynchronize(0);
}

void Stream::record(bool external) {
	//if (ptr_ == 0)
	//	return;
	CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
	// cudaEventDestroy(ev);
	// cudaEventCreate(&ev, cudaEventDisableTiming);
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
	updated = true;
#endif
}

void Stream::wait_recorded(const Stream& s) {
	if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
		return;
#if !DISABLE_STREAMS
	assert(s.updated);
	cudaStreamWaitEvent(ptr_, s.ev);
	this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	if (/*ptr_ == 0 ||*/ s.ptr_ == 0 || ptr_ == s.ptr_) {
		updated = false;
		return;
	}
	//assert(ptr_ != nullptr);
	//assert(s.ptr_ != nullptr);
	assert(s.ev != nullptr);
	if (!s.updated) {
		assert(!external); // Has to be recorded in the origin graph
		CudaCheckErrorModNoSync;
		cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
		s.updated = true;
		CudaCheckErrorModNoSync;
	}
	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
	this->updated = false;
#endif
	CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	if (s == 0 || s == ptr_) {
		updated = false;
		return;
	}
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	CudaCheckErrorModNoSync;
	cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
	updated = false;
	CudaCheckErrorModNoSync;

	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(ptr_, ev, cudaEventWaitDefault);
#endif
	CudaCheckErrorModNoSync;
}

int low  = -1;
int high = -1;

constexpr int POOL_SIZE = 37;
cudaStream_t stream_pool[MAXG][POOL_SIZE];

bool initPool() {
	int devs;
	cudaGetDeviceCount(&devs);
	for (int j = 0; j < devs; j++) {
		cudaSetDevice(j);
		for (int i = 0; i < POOL_SIZE; ++i) {
			cudaStreamCreateWithFlags(&stream_pool[j][i], 0);
		}
	}
	return true;
}

#define USEPOOL true
#if USEPOOL
bool initialized = initPool();
#else
bool initialized = false;
#endif

void Stream::init(int priority) {
	static int assigner_idx[MAXD] = { 0 };
	if (ptr_) {
		// free[ptr]++;
		cudaEventDestroy(ev);
		// cudaStreamDestroy(ptr_);
		ptr_ = nullptr;
		ev   = nullptr;
	}

#if !DISABLE_STREAMS
	if (high == -1) {
		cudaDeviceGetStreamPriorityRange(&low, &high);
	}

	// int prio = low + priority * ((high - low - 1)) / 100;

#if USEPOOL
	int dev;
	cudaGetDevice(&dev);
	ptr_ = stream_pool[dev][assigner_idx[dev]];
	assigner_idx[dev] += 1;
	if (assigner_idx[dev] >= POOL_SIZE)
		assigner_idx[dev] -= POOL_SIZE;
#else
	cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, priority);
	// cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
#endif

	cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
	cudaEventCreate(&ev, cudaEventDisableTiming);
#else
	ptr_ = nullptr;
	ev   = nullptr;
#endif
	// free[ptr] = 0;
}

void Stream::initDefault() {
	ptr_    = 0;
	ev      = nullptr;
	updated = true;
}

// std::map<void *, int> free;

Stream::~Stream() {
	if (ptr_) {
		// free[ptr]++;
		// cudaStreamDestroy(ptr_);
		ptr_ = nullptr;
	}
	if (ev) {
		cudaEventDestroy(ev);
		ev = nullptr;
	}
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept
	: ptr_(s.ptr_), ev(s.ev) {
	s.ptr_ = nullptr;
	s.ev   = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
	if (GPUprop.empty()) {
		int count = 0;
		cudaGetDeviceCount(&count);
		for (int i = 0; i < count; ++i) {
			GPUprop.emplace_back();
			cudaGetDeviceProperties(&GPUprop.back(), i);

			std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
				<< ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
				<< ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
				<< " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

				<< "-bit" << std::endl;
		}
	}
}

std::mutex mempool_lock[MAXG];

std::map<int, std::vector<void*>> size_to_memory[MAXG];

// GPUtrim needs to give whole slabs back to the driver, but the free list above is a flat
// list of interchangeable void* slices with no record of which cudaMallocAsync slab each
// one came from -- and only a slab's original base pointer is a valid cudaFreeAsync
// argument, never a mid-slab slice. So, non-invasively, we remember every slab we carve
// (its base + full size) here at allocation time and leave the malloc/free hot path
// otherwise untouched. Pointers that end up in a bucket without belonging to any recorded
// slab (e.g. a foreign cudaMalloc'd buffer freed with bytes<64K, which the coercion below
// files into the 1024-byte bucket) simply fall inside no slab and are never trimmed --
// exactly the original behavior.
struct GPUSlab {
	void* base;
	size_t bytes; // full slab size
};
std::map<int, std::vector<GPUSlab>> slab_registry[MAXG];

FIDESlib::Stream s[MAXG];

#define MEMPOOL true
// void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
	void* ptr = nullptr;

	uint64_t MBs = 1024;

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
		MBs   = bytes / 1024;
	}
#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0) {
		std::vector<void*>& free_limb = size_to_memory[id][bytes];

		if (s[id].ptr() == nullptr) {
			mempool_lock[id].lock();
			if (s[id].ptr() == nullptr) {
				s[id].init();
			}
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;
		if (free_limb.empty()) {
			uint64_t* base;
			cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());

			mempool_lock[id].lock();
			slab_registry[id][bytes].push_back(GPUSlab{ (void*)base, MBs * 1024 * 1024 });
			for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes) {
				free_limb.emplace_back(((char*)base) + i);
			}
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;

		//if (stream != nullptr) {
		s[id].record();
		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(stream, s[id].ev);
		//}

		CudaCheckErrorModNoSync;
		mempool_lock[id].lock();
		ptr = free_limb.back();
		free_limb.pop_back();
		mempool_lock[id].unlock();
		// ptr = free_limb.front();
		// free_limb.pop_front();
		// std::cout << "get " << ptr << std::endl;
		return ptr;
	}

#endif
	// std::cout << bytes << std::endl;
	cudaMallocAsync(&ptr, bytes, 0);

	return ptr;
}

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
	}

#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0) {
		std::vector<void*>& free_limb = size_to_memory[id][bytes];
		if (s[id].ptr() == nullptr) {
			mempool_lock[id].lock();
			s[id].init();
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;
		// cudaDeviceSynchronize();
		//if (stream != nullptr) {
		s[id].wait(stream);
		//}
		CudaCheckErrorModNoSync;
		mempool_lock[id].lock();
		free_limb.emplace_back(ptr);
		mempool_lock[id].unlock();
		// std::cout << "free " << ptr << std::endl;
		return;
	}
#endif

	cudaFreeAsync(ptr, stream);
}

void GPUtrim(int id) {
	cudaSetDevice(id);
	if (s[id].ptr() == nullptr) {
		s[id].init();
	}
	// Slices were pushed back onto the free list only after waiting on their owning stream
	// (see GPUfree), but that wait is stream-ordered, not host-blocking: sync here so the
	// cudaFreeAsync below cannot race a kernel still using a slab we are about to release.
	cudaStreamSynchronize(s[id].ptr());

	mempool_lock[id].lock();
	for (auto& [bytes, slabs] : slab_registry[id]) {
		std::vector<void*>& free_limb = size_to_memory[id][bytes];
		if (slabs.empty()) {
			continue;
		}

		// A slab is fully idle iff every slice we carved from it is back in the free list.
		// Count the free slices that fall inside each slab; foreign pointers match no slab.
		std::vector<size_t> freeCount(slabs.size(), 0);
		for (void* p : free_limb) {
			for (size_t i = 0; i < slabs.size(); ++i) {
				char* base = (char*)slabs[i].base;
				if ((char*)p >= base && (char*)p < base + slabs[i].bytes) {
					freeCount[i]++;
					break;
				}
			}
		}

		std::vector<bool> freed(slabs.size(), false);
		bool any = false;
		for (size_t i = 0; i < slabs.size(); ++i) {
			if (freeCount[i] == slabs[i].bytes / bytes) {
				cudaFreeAsync(slabs[i].base, s[id].ptr());
				freed[i] = true;
				any		 = true;
			}
		}
		if (!any) {
			continue;
		}

		// Drop the freed slabs' slices from the free list and the slabs from the registry.
		std::vector<void*> survivingChunks;
		survivingChunks.reserve(free_limb.size());
		for (void* p : free_limb) {
			bool drop = false;
			for (size_t i = 0; i < slabs.size(); ++i) {
				if (!freed[i]) {
					continue;
				}
				char* base = (char*)slabs[i].base;
				if ((char*)p >= base && (char*)p < base + slabs[i].bytes) {
					drop = true;
					break;
				}
			}
			if (!drop) {
				survivingChunks.push_back(p);
			}
		}
		free_limb = std::move(survivingChunks);

		std::vector<GPUSlab> survivingSlabs;
		for (size_t i = 0; i < slabs.size(); ++i) {
			if (!freed[i]) {
				survivingSlabs.push_back(slabs[i]);
			}
		}
		slabs = std::move(survivingSlabs);
	}
	mempool_lock[id].unlock();

	// The cudaFreeAsync calls above only return slabs to the async-malloc pool; because the
	// pool's release threshold is pinned to UINT64_MAX at context creation (see
	// ContextData::ContextData), the pool caches that memory instead of handing it back to
	// the OS -- so cudaMemGetInfo/nvidia-smi would still report it as used. Drain the frees,
	// then explicitly trim the pool to 0 bytes reserved so the now-idle slabs actually
	// become free VRAM the rest of the system can use. Live allocations are untouched.
	cudaStreamSynchronize(s[id].ptr());
	cudaMemPool_t mp;
	cudaDeviceGetDefaultMemPool(&mp, id);
	cudaMemPoolTrimTo(mp, 0);
}

int GetTargetThreads(int id) {
	return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}

} // namespace FIDESlib