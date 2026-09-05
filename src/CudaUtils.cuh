//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cuda.h>
#include <driver_types.h>
#include <execinfo.h>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace FIDESlib {

void initGPUprop();
int GetTargetThreads(int id);

enum NVTX_CATEGORIES { NONE, LIFETIME, FUNCTION };

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0);
void CudaNvtxStop(const std::string msg = "", NVTX_CATEGORIES cat = FUNCTION);

class CudaNvtxRange {
	const std::string msg;
	const NVTX_CATEGORIES cat;
	bool valid = true;

  public:
	explicit CudaNvtxRange(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0) : msg(msg), cat(cat) {
		CudaNvtxStart(msg, cat, val);
	}

	CudaNvtxRange(CudaNvtxRange&& r) noexcept : msg(r.msg), cat(r.cat) {
		this->valid = r.valid;
		r.valid		= false;
	}

	~CudaNvtxRange() {
		if (valid)
			CudaNvtxStop(msg, cat);
	}
};

int getNumDevices();

void CudaHostSync();

inline void breakpoint() {
}

// TODO: Remove the cudart unloading.
#define CudaCheckErrorMod                                                                    \
	do {                                                                                     \
		cudaDeviceSynchronize();                                                             \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e == cudaErrorCudartUnloading) {                                                 \
			exit(0);                                                                         \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {             \
                                                                                             \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

#define CudaCheckErrorModMGPU                                                                \
	do {                                                                                     \
		cudaStreamSynchronize(0);                                                            \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {                    \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

// TODO: FIX THE CUDARTUNLOADING ERROR, IT HAPPENS WHEN THE LIBRARY IS BEING UNLOADED, CAN BE IGNORED FOR NOW
#define CudaCheckErrorModNoSync                                                                                          \
	do {                                                                                                                 \
		/*cudaDeviceSynchronize();*/                                                                                     \
		cudaError_t e = cudaGetLastError();                                                                              \
		if (e == cudaErrorCudartUnloading) {                                                                             \
			exit(0);                                                                                                     \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && e != cudaErrorGraphExecUpdateFailure) { \
			void* array[10];                                                                                             \
			size_t size;                                                                                                 \
			size = backtrace(array, 10);                                                                                 \
			backtrace_symbols_fd(array, size, STDERR_FILENO);                                                            \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                             \
			FIDESlib::breakpoint();                                                                                      \
			exit(0);                                                                                                     \
		}                                                                                                                \
	} while (0)

#define NCCLCHECK(cmd)                                                                              \
	do {                                                                                            \
		ncclResult_t res = cmd;                                                                     \
		if (res != ncclSuccess) {                                                                   \
			printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
			exit(EXIT_FAILURE);                                                                     \
		}                                                                                           \
	} while (0)

class Event;

extern std::map<void*, int> free;

class Stream {
  private:
	cudaStream_t ptr_ = nullptr;

  public:
	cudaEvent_t ev = nullptr;
	bool updated   = false;
	// Event ev;

	void init(int priority = 0);

	cudaStream_t ptr() {
		updated = false;
		return ptr_;
	}

	void initDefault();

	// void wait(const Event &ev) const;
	void wait(Stream& s, bool external = false);
	void wait(cudaStream_t s);

	Stream();

	Stream(Stream& s) = delete;

	Stream(const Stream& s) = delete;

	Stream& operator=(const Stream&) = delete;

	Stream(Stream&& s) noexcept;

	~Stream();

	void record(bool external = false);

	void wait_recorded(const Stream& s);

	void capture_begin();

	void capture_end();
};

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/// Allocate from / return to the process-local caching pool. `device` is the CUDA device
/// ordinal, and it selects the pool: the free lists, their slabs and the pool's own stream and
/// event are all per physical device. Passing anything else -- a logical index into a context's
/// GPUid list, say -- both strands the chunk in the wrong pool and lets a later allocation on
/// that ordinal be served memory that lives on another device.
void* GPUmalloc(int device, int bytes, cudaStream_t stream, bool cache = false);
void GPUfree(void* ptr, int device, int bytes, cudaStream_t stream, bool cache = false);

/// @brief Release every buffer currently sitting in GPUalloc/GPUfree's process-local
/// caching pool for device `id` back to the CUDA driver. GPUfree never does this on its
/// own (buffers are kept around for cheap reuse), so a caller that actually wants VRAM
/// back after freeing a batch of large objects (e.g. offloaded ciphertexts) must call
/// this explicitly. Synchronous: blocks until the device's internal stream drains before
/// freeing, so avoid calling it on a hot path.
void GPUtrim(int device);

struct GPUPoolBucketStats {
	size_t chunkBytes;
	size_t slabCount;
	size_t reservedBytes;
	size_t totalChunks;
	size_t freeChunks;
	size_t liveChunks;
};

/// Synchronous diagnostic snapshot of FIDESlib's process-local caching allocator.
/// Intended for profiling only; it does not allocate, free, or trim any buffer.
std::vector<GPUPoolBucketStats> GPUmemoryPoolStats(int device);

} // namespace FIDESlib
#endif // FIDESLIB_CUDAUTILS_CUH
