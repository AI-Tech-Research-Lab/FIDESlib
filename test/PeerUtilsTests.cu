//
// Created by carlosad on 18/12/25.
//

#include <errno.h>
#ifdef NCCL
#include <nccl.h>
#endif

#include <iomanip>

#include <CudaUtils.cuh>
#include <PeerUtils.cuh>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include <omp.h>
// Simple CUDA error check macro
#define CUDA_CHECK(stmt)                                                                                                  \
	do {                                                                                                                  \
		cudaError_t err = (stmt);                                                                                         \
		if (err != cudaSuccess) {                                                                                         \
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
			FIDESlib::breakpoint();                                                                                       \
		}                                                                                                                 \
	} while (0)

// Helper: initialize host data
static void init_host_data(std::vector<float>& h, float seed) {
	for (size_t i = 0; i < h.size(); ++i) {
		h[i] = seed + static_cast<float>(i) * 0.001f;
	}
}

// Helper: compare host buffers
static void expect_equal(const std::vector<float>& a, const std::vector<float>& b) {
	ASSERT_EQ(a.size(), b.size());
	for (size_t i = 0; i < a.size(); ++i) {
		ASSERT_FLOAT_EQ(a[i], b[i]) << "mismatch at index " << i;
	}
}

/**
 * Enable P2P access safely: handle case where already enabled
 * Returns true if P2P is enabled (either was already or just enabled)
 * Returns false if P2P cannot be enabled
 */
static void safe_enable_p2p(int src_gpu, int dst_gpu) {
	int can_access = 0;
	CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_gpu, dst_gpu));

	if (!can_access) {
		std::cout << "  P2P not available: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	}

	CUDA_CHECK(cudaSetDevice(src_gpu));

	// Try to enable P2P; if already enabled, cudaDeviceEnablePeerAccess
	// returns cudaErrorPeerAccessAlreadyEnabled, which we ignore
	cudaError_t err = cudaDeviceEnablePeerAccess(dst_gpu, 0);

	if (err == cudaErrorPeerAccessAlreadyEnabled) {
		std::cout << "  P2P already enabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	} else if (err == cudaSuccess) {
		std::cout << "  P2P enabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
		return;
	} else {
		std::cerr << "  Failed to enable P2P: GPU " << src_gpu << " -> GPU " << dst_gpu << ": " << cudaGetErrorString(err) << std::endl;
		return;
	}
}

/**
 * Disable P2P access safely: handle case where not enabled
 */
[[maybe_unused]] static void safe_disable_p2p(int src_gpu, int dst_gpu) {
	CUDA_CHECK(cudaSetDevice(src_gpu));
	cudaError_t err = cudaDeviceDisablePeerAccess(dst_gpu);

	if (err == cudaErrorPeerAccessNotEnabled) {
		// Already disabled, that's fine
		return;
	} else if (err == cudaSuccess) {
		std::cout << "  P2P disabled: GPU " << src_gpu << " -> GPU " << dst_gpu << std::endl;
	} else {
		std::cerr << "  Warning: Failed to disable P2P: GPU " << src_gpu << " -> GPU " << dst_gpu << ": " << cudaGetErrorString(err) << std::endl;
	}
}

// Common test body: assumes src/dst are on devDst and devSrc chosen appropriately
static void run_p2p_test(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(cudaSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(cudaSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));

	uint32_t* d_flag = nullptr;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	// Launch transfer and notify kernels on src device
	CUDA_CHECK(cudaSetDevice(devSrc));
	cudaStream_t sSrc;
	CUDA_CHECK(cudaStreamCreate(&sSrc));

	int threads = 256;
	int blocks	= static_cast<int>((n + threads - 1) / threads);

	FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
	CUDA_CHECK(cudaGetLastError());

	uint32_t expectedValue = 1;
	FIDESlib::notify_kernel<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
	CUDA_CHECK(cudaGetLastError());

	// On destination device, wait for flag and then read back
	CUDA_CHECK(cudaSetDevice(devDst));
	cudaStream_t sDst;
	CUDA_CHECK(cudaStreamCreate(&sDst));

	FIDESlib::p2p_polling_kernel<<<1, 256, 0, sDst>>>(d_flag, expectedValue);
	CUDA_CHECK(cudaGetLastError());

	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(cudaStreamSynchronize(sDst));

	// Copy dst back to host from destination device
	CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(cudaStreamDestroy(sSrc));
	CUDA_CHECK(cudaStreamDestroy(sDst));

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaFree(d_src));

	CUDA_CHECK(cudaSetDevice(devDst));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_flag));
}

static void run_p2p_test_graph(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(cudaSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(cudaSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));

	FIDESlib::TimelineSemaphore* d_flag = nullptr;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	// Launch transfer and notify kernels on src device
	CUDA_CHECK(cudaSetDevice(devSrc));
	cudaStream_t sSrc;
	CUDA_CHECK(cudaStreamCreate(&sSrc));

	CUDA_CHECK(cudaStreamBeginCapture(sSrc, cudaStreamCaptureModeGlobal));

	CUDA_CHECK(cudaSetDevice(devDst));
	cudaStream_t sDst;
	CUDA_CHECK(cudaStreamCreate(&sDst));

	CUDA_CHECK(cudaSetDevice(devSrc));

	cudaEvent_t event0;
	CUDA_CHECK(cudaEventCreateWithFlags(&event0, cudaEventDisableTiming));
	CUDA_CHECK(cudaEventRecord(event0, sSrc));
	CUDA_CHECK(cudaStreamWaitEvent(sDst, event0));

	CUDA_CHECK(cudaEventDestroy(event0));

	if (0) {
		int threads = 256;
		int blocks	= static_cast<int>((n + threads - 1) / threads);

		FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
	} else {
		FIDESlib::transferKernel(d_src, d_dst, n, sSrc, devSrc, devDst);
	}
	CUDA_CHECK(cudaGetLastError());

	uint32_t expectedValue = 1;

	if (0) {
		FIDESlib::notify_kernel_hostpin<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
	} else {
		FIDESlib::notifyKernel(d_flag, expectedValue, sSrc);
	}
	CUDA_CHECK(cudaGetLastError());

	// On destination device, wait for flag and then read back

	if (0) {
		FIDESlib::hostpin_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, expectedValue);
	} else {
		FIDESlib::pollingKernel(d_flag, expectedValue, sDst);
	}

	CUDA_CHECK(cudaGetLastError());

	cudaEvent_t event1;

	CUDA_CHECK(cudaSetDevice(devDst));
	CUDA_CHECK(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
	CUDA_CHECK(cudaEventRecord(event1, sDst));
	CUDA_CHECK(cudaStreamWaitEvent(sSrc, event1));

	CUDA_CHECK(cudaEventDestroy(event1));
	{
		cudaGraph_t graph;
		CUDA_CHECK(cudaStreamEndCapture(sSrc, &graph));

		cudaGraphExec_t exec;
		CUDA_CHECK(cudaGraphInstantiate(&exec, graph));

		CUDA_CHECK(cudaGraphLaunch(exec, sSrc));
		CudaCheckErrorMod;
		CUDA_CHECK(cudaGraphDestroy(graph));
		CUDA_CHECK(cudaGraphExecDestroy(exec));
		CudaCheckErrorMod;
	}

	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(cudaStreamSynchronize(sDst));
	CUDA_CHECK(cudaStreamSynchronize(sSrc));
	// Copy dst back to host from destination device
	CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(cudaStreamDestroy(sSrc));
	CUDA_CHECK(cudaStreamDestroy(sDst));

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaFree(d_src));

	CUDA_CHECK(cudaSetDevice(devDst));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_flag));
}

static void run_p2p_test_graph_parallel(int devSrc, int devDst, size_t n) {
	CUDA_CHECK(cudaSetDevice(devSrc));
	float* d_src = nullptr;
	CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));

	CUDA_CHECK(cudaSetDevice(devDst));
	float* d_dst = nullptr;
	CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));

	FIDESlib::TimelineSemaphore* d_flag = nullptr;
	CUDA_CHECK(cudaMalloc(&d_flag, sizeof(uint32_t)));
	CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(uint32_t)));

	// Initialize host data and copy to src device
	std::vector<float> h_src(n), h_dst(n);
	init_host_data(h_src, 1.0f);

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	uint32_t expectedValue = 1;

	CUDA_CHECK(cudaSetDevice(devSrc));
	cudaStream_t sSrc;
	CUDA_CHECK(cudaStreamCreate(&sSrc));
	CUDA_CHECK(cudaSetDevice(devDst));
	cudaStream_t sDst;
	CUDA_CHECK(cudaStreamCreate(&sDst));

#pragma omp parallel num_threads(2)
	{
		int j = omp_get_thread_num();

		if (j == 1) {
			// Launch transfer and notify kernels on src device
			CUDA_CHECK(cudaSetDevice(devSrc));

			CUDA_CHECK(cudaStreamBeginCapture(sSrc, cudaStreamCaptureModeThreadLocal));

			CUDA_CHECK(cudaSetDevice(devSrc));

			if (0) {
				int threads = 256;
				int blocks	= static_cast<int>((n + threads - 1) / threads);

				FIDESlib::p2p_transfer_1d<<<blocks, threads, 0, sSrc>>>(d_src, d_dst, n);
			} else {
				FIDESlib::transferKernel(d_src, d_dst, n, sSrc, devSrc, devDst);
			}
			CUDA_CHECK(cudaGetLastError());

			if (0) {
				FIDESlib::notify_kernel_hostpin<<<1, 32, 0, sSrc>>>(d_flag, expectedValue);
			} else {
				FIDESlib::notifyKernel(d_flag, expectedValue, sSrc);
			}
			CUDA_CHECK(cudaGetLastError());

			{
				cudaGraph_t graph;
				CUDA_CHECK(cudaStreamEndCapture(sSrc, &graph));

				cudaGraphExec_t exec;
				CUDA_CHECK(cudaGraphInstantiate(&exec, graph));

				CUDA_CHECK(cudaGraphLaunch(exec, sSrc));
				CudaCheckErrorMod;
				CUDA_CHECK(cudaGraphDestroy(graph));
				CUDA_CHECK(cudaGraphExecDestroy(exec));
				CudaCheckErrorMod;
			}
		} else {
			CUDA_CHECK(cudaSetDevice(devDst));
			// On destination device, wait for flag and then read back
			CUDA_CHECK(cudaStreamBeginCapture(sDst, cudaStreamCaptureModeThreadLocal));

			if (0) {
				FIDESlib::hostpin_polling_kernel<<<1, 32, 0, sDst>>>(d_flag, expectedValue);
			} else {
				FIDESlib::pollingKernel(d_flag, expectedValue, sDst);
			}

			CUDA_CHECK(cudaGetLastError());

			{
				cudaGraph_t graph;
				CUDA_CHECK(cudaStreamEndCapture(sDst, &graph));

				cudaGraphExec_t exec;
				CUDA_CHECK(cudaGraphInstantiate(&exec, graph));

				CUDA_CHECK(cudaGraphLaunch(exec, sDst));
				CudaCheckErrorMod;
				CUDA_CHECK(cudaGraphDestroy(graph));
				CUDA_CHECK(cudaGraphExecDestroy(exec));
				CudaCheckErrorMod;
			}
		}
	}
	// Wait for destination stream to finish (i.e., transfer+notify complete)
	CUDA_CHECK(cudaStreamSynchronize(sDst));
	CUDA_CHECK(cudaStreamSynchronize(sSrc));
	// Copy dst back to host from destination device
	CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

	// Verify correctness
	expect_equal(h_src, h_dst);

	// Cleanup
	CUDA_CHECK(cudaStreamDestroy(sSrc));
	CUDA_CHECK(cudaStreamDestroy(sDst));

	CUDA_CHECK(cudaSetDevice(devSrc));
	CUDA_CHECK(cudaFree(d_src));

	CUDA_CHECK(cudaSetDevice(devDst));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_flag));
}

TEST(P2PTransferTest, SingleGPU) {
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	ASSERT_GE(deviceCount, 1) << "Need at least 1 GPU";

	int dev	 = 0;
	size_t n = 1 << 20; // 1M floats

	run_p2p_test(dev, dev, n);
}

TEST(P2PTransferTest, SingleGPUGraph) {
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	ASSERT_GE(deviceCount, 1) << "Need at least 1 GPU";

	int dev	 = 0;
	size_t n = 1 << 20; // 1M floats

	run_p2p_test_graph(dev, dev, n);
}

// 2. Two-GPU test: devSrc != devDst, with P2P if available
TEST(P2PTransferTest, MultiGPUIfAvailable) {
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test(devSrc, devDst, n);
}

TEST(P2PTransferTest, MultiGPUIfAvailableSingleGraph) {
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test_graph(devSrc, devDst, n);
}

TEST(P2PTransferTest, MultiGPUIfAvailableParallelGraphs) {
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (deviceCount < 2) {
		GTEST_SKIP() << "Less than 2 GPUs; skipping multi-GPU test";
	}

	int devSrc = 0;
	int devDst = 1;

	int canAccessSrcToDst = 0, canAccessDstToSrc = 0;
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessSrcToDst, devSrc, devDst));
	CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessDstToSrc, devDst, devSrc));

	if (!canAccessSrcToDst || !canAccessDstToSrc) {
		GTEST_SKIP() << "P2P not available between GPU " << devSrc << " and GPU " << devDst << "; skipping";
	}

	std::cout << "\nEnabling P2P access..." << std::endl;
	safe_enable_p2p(devSrc, devDst);
	safe_enable_p2p(devDst, devSrc);

	size_t n = 1 << 20; // 1M floats
	run_p2p_test_graph_parallel(devSrc, devDst, n);
}
