//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_VECTORGPU_CUH
#define FIDESLIB_VECTORGPU_CUH
#include "CudaUtils.cuh"
#include "LogUtil.hpp"
#include "cuda.h"
#include <cassert>
#include <driver_types.h>
#include <type_traits>
#include <vector>

namespace FIDESlib {
template <typename T> class VectorGPU {
	static_assert(std::is_trivially_copyable_v<T>);
	bool freeing;
	bool managed;

  public:
	T* data;
	const int size;
	const int device;

	VectorGPU(VectorGPU<T>&& v) noexcept;
	/*
		VectorGPU<T> &operator=(VectorGPU<T> && v) noexcept
		 {
			freeing = v.freeing;
			managed = v.managed;
			data = v.data;
			size = v.size;
			device = v.device;
			v.freeing = true;
			return *this;
		}
*/
	VectorGPU<T>& operator=(VectorGPU<T>& other)   = delete;
	VectorGPU<T>& operator=(const VectorGPU<T>& v) = delete;
	VectorGPU(const VectorGPU<T>& v)			   = delete;
	VectorGPU(T* data, const int size, const int device, const int offset = 0);
	VectorGPU(Stream& stream, const int size, const int device, const T* src = nullptr);
	/** Device bytes this vector owns, i.e. that its own free()/destructor releases. Zero for
	 * an unmanaged vector -- a view into someone else's buffer (see the T* constructor) --
	 * so that summing over a set of vectors never counts the same allocation twice. */
	[[nodiscard]] size_t ownedBytes() const { return managed ? static_cast<size_t>(size) * sizeof(T) : 0; }
	void free(Stream& stream);
	~VectorGPU();
};

} // namespace FIDESlib
#endif // FIDESLIB_VECTORGPU_CUH
