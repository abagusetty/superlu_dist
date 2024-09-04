/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * \brief Wrappers for multiple types of GPUs
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * May 22, 2022
 * </pre>
 */

#ifndef __SUPERLU_GPUWRAPPER /* allow multiple inclusions */
#define __SUPERLU_GPUWRAPPER

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuda_profiler_api.h>

#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuHostMalloc cudaHostAlloc
#define gpuHostMallocDefault cudaHostAllocDefault
#define gpuMallocManaged cudaMallocManaged
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpy2DAsync cudaMemcpy2DAsync
#define gpuMemcpy2D cudaMemcpy2D
#define gpuFreeHost cudaFreeHost
#define gpuFree cudaFree
#define gpuMemPrefetchAsync cudaMemPrefetchAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuMemAttachGlobal cudaMemAttachGlobal
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroyWithFlags cudaStreamDestroyWithFlags
#define gpuStreamDefault cudaStreamDefault
#define gpublasStatus_t cublasStatus_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuMemGetInfo cudaMemGetInfo
#define gpuOccupancyMaxPotentialBlockSize cudaOccupancyMaxPotentialBlockSize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceReset cudaDeviceReset
#define gpuMallocHost cudaMallocHost
#define gpuEvent_t cudaEvent_t
#define gpuMemset cudaMemset
#define  GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define  GPUBLAS_STATUS_NOT_INITIALIZED CUBLAS_STATUS_NOT_INITIALIZED
#define  GPUBLAS_STATUS_ALLOC_FAILED CUBLAS_STATUS_ALLOC_FAILED
#define  GPUBLAS_STATUS_INVALID_VALUE CUBLAS_STATUS_INVALID_VALUE
#define  GPUBLAS_STATUS_ARCH_MISMATCH CUBLAS_STATUS_ARCH_MISMATCH
#define  GPUBLAS_STATUS_MAPPING_ERROR CUBLAS_STATUS_MAPPING_ERROR
#define  GPUBLAS_STATUS_EXECUTION_FAILED CUBLAS_STATUS_EXECUTION_FAILED
#define  GPUBLAS_STATUS_INTERNAL_ERROR CUBLAS_STATUS_INTERNAL_ERROR
#define  GPUBLAS_STATUS_LICENSE_ERROR CUBLAS_STATUS_LICENSE_ERROR
#define  GPUBLAS_STATUS_NOT_SUPPORTED CUBLAS_STATUS_NOT_SUPPORTED
#define  gpublasCreate cublasCreate
#define  gpublasDestroy cublasDestroy
#define  gpublasHandle_t cublasHandle_t
#define  gpublasSetStream cublasSetStream
#define  gpublasDgemm cublasDgemm
#define  gpublasSgemm cublasSgemm
#define  gpublasZgemm cublasZgemm
#define  gpublasCgemm cublasCgemm
#define  gpublasSideMode_t cublasSideMode_t
#define  gpublasFillMode_t cublasFillMode_t
#define  gpublasDiagType_t cublasDiagType_t
#define  gpublasOperation_t cublasOperation_t
#define  GPUBLAS_OP_N CUBLAS_OP_N
#define  GPUBLAS_DIAG_UNIT CUBLAS_DIAG_UNIT
#define  GPUBLAS_DIAG_NON_UNIT CUBLAS_DIAG_NON_UNIT
#define  GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define  GPUBLAS_SIDE_RIGHT CUBLAS_SIDE_RIGHT
#define  GPUBLAS_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER
#define  GPUBLAS_FILL_MODE_UPPER CUBLAS_FILL_MODE_UPPER
#define  gpusolverStatus_t cusolverStatus_t
#define  GPUSOLVER_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
#define  gpusolverDnHandle_t cusolverDnHandle_t
#define  gpusolverDnCreate cusolverDnCreate
#define  gpusolverDnSetStream cusolverDnSetStream
#define  gpusolverDnDgetrf cusolverDnDgetrf
#define  gpusolverDnDgetrf_bufferSize cusolverDnDgetrf_bufferSize
#define  gpusolverDnDestroy cusolverDnDestroy
#define  gpusolverDnDgetrf cusolverDnDgetrf
#define  gpusolverDnSgetrf cusolverDnSgetrf
#define  gpusolverDnCgetrf cusolverDnCgetrf
#define  gpusolverDnZgetrf cusolverDnZgetrf
#define  gpublasDtrsm cublasDtrsm
#define  gpublasSscal cublasSscal
#define  gpublasDscal cublasDscal
#define  gpublasCscal cublasCscal
#define  gpublasZscal cublasZscal
#define  gpublasSaxpy cublasSaxpy
#define  gpublasDaxpy cublasDaxpy
#define  gpublasCaxpy cublasCaxpy
#define  gpublasZaxpy cublasZaxpy
#define  gpuComplex cuComplex
#define  gpuDoubleComplex cuDoubleComplex
#define  gpuRuntimeGetVersion cudaRuntimeGetVersion
#define  gpuGetLastError cudaGetLastError
#define  threadIdx_x threadIdx.x
#define  threadIdx_y threadIdx.y
#define  blockIdx_x blockIdx.x
#define  blockIdx_y blockIdx.y
#define  blockIdx_z blockIdx.z
#define  blockDim_x blockDim.x
#define  blockDim_y blockDim.y
#define  gridDim_x gridDim.x
#define  gridDim_y gridDim.y
#define  gputhrust thrust
#define  gputhrust_policy thrust::system::cuda::par
#define  gputhrust_device_ptr thrust::device_ptr

#define gpublasCheckErrors(fn)                  \
	 do { \
		 gpublasStatus_t __err = fn; \
		 if (__err != GPUBLAS_STATUS_SUCCESS) { \
			 fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
				 (int)(__err), \
				 __FILE__, __LINE__); \
			 fprintf(stderr, "*** FAILED - ABORTING\n"); \
			 exit(1); \
		 } \
	 } while(0);

#define gpusolverCheckErrors(fn)                  \
	 do { \
		 gpusolverStatus_t __err = fn; \
		 if (__err != GPUSOLVER_STATUS_SUCCESS) { \
			 fprintf(stderr, "Fatal cusolver error: %d (at %s:%d)\n", \
				 (int)(__err), \
				 __FILE__, __LINE__); \
			 fprintf(stderr, "*** FAILED - ABORTING\n"); \
			 exit(1); \
		 } \
	 } while(0);


#elif defined(HAVE_HIP)

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

// #include "roctracer_ext.h"    // need to pass the include dir directly to HIP_HIPCC_FLAGS
// // roctx header file
// #include <roctx.h>

#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuHostMalloc hipHostMalloc
#define gpuHostMallocDefault hipHostMallocDefault
#define gpuMallocManaged hipMallocManaged
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpy2DAsync hipMemcpy2DAsync
#define gpuMemcpy2D hipMemcpy2D
#define gpuFreeHost hipHostFree
#define gpuFree hipFree
#define gpuMemPrefetchAsync hipMemPrefetchAsync   // not sure about this
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuMemAttachGlobal hipMemAttachGlobal
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroyWithFlags hipStreamDestroyWithFlags
#define gpuStreamDefault hipStreamDefault
#define gpublasStatus_t hipblasStatus_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuMemGetInfo hipMemGetInfo
#define gpuOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceReset hipDeviceReset
#define gpuMallocHost hipHostMalloc
#define gpuEvent_t hipEvent_t
#define gpuMemset hipMemset
#define  GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define  GPUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define  GPUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define  GPUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define  GPUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define  GPUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define  GPUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define  GPUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define  GPUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_LICENSE_ERROR
#define  GPUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define  gpublasCreate hipblasCreate
#define  gpublasDestroy hipblasDestroy
#define  gpublasHandle_t hipblasHandle_t
#define  gpublasSetStream hipblasSetStream
#define  gpublasSideMode_t hipblasSideMode_t
#define  gpublasFillMode_t hipblasFillMode_t
#define  gpublasDiagType_t hipblasDiagType_t
#define  gpublasOperation_t hipblasOperation_t
#define  gpublasDgemm hipblasDgemm
#define  gpublasSgemm hipblasSgemm
#define  gpublasZgemm hipblasZgemm
#define  gpublasCgemm hipblasCgemm
#define  GPUBLAS_OP_N HIPBLAS_OP_N
#define  GPUBLAS_DIAG_UNIT HIPBLAS_DIAG_UNIT
#define  GPUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define  GPUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define  GPUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define  GPUBLAS_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER
#define  GPUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define  gpusolverStatus_t hipsolverStatus_t
#define  GPUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define  gpusolverDnHandle_t hipsolverHandle_t
#define  gpusolverDnCreate hipsolverDnCreate
#define  gpusolverDnSetStream hipsolverDnSetStream
#define  gpusolverDnDgetrf hipsolverDnDgetrf
#define  gpusolverDnDgetrf_bufferSize hipsolverDnDgetrf_bufferSize
#define  gpusolverDnDestroy hipsolverDnDestroy
#define  gpusolverDnDgetrf hipsolverDnDgetrf
#define  gpusolverDnSgetrf hipsolverDnSgetrf
#define  gpusolverDnCgetrf hipsolverDnCgetrf
#define  gpusolverDnZgetrf hipsolverDnZgetrf
#define  gpublasDtrsm hipblasDtrsm
#define  gpublasSscal hipblasSscal
#define  gpublasDscal hipblasDscal
#define  gpublasCscal hipblasCscal
#define  gpublasZscal hipblasZscal
#define  gpublasSaxpy hipblasSaxpy
#define  gpublasDaxpy hipblasDaxpy
#define  gpublasCaxpy hipblasCaxpy
#define  gpublasZaxpy hipblasZaxpy
#define  gpuComplex hipblasComplex
#define  gpuDoubleComplex hipblasDoubleComplex
#define  gpuRuntimeGetVersion hipRuntimeGetVersion
#define  gpuGetLastError hipGetLastError
#define  threadIdx_x hipThreadIdx_x
#define  threadIdx_y hipThreadIdx_y
#define  blockIdx_x hipBlockIdx_x
#define  blockIdx_y hipBlockIdx_y
#define  blockIdx_z hipBlockIdx_z
#define  blockDim_x hipBlockDim_x
#define  blockDim_y hipBlockDim_y
#define  gridDim_x hipGridDim_x
#define  gridDim_y hipGridDim_y
#define  gputhrust thrust
#define  gputhrust_policy thrust::system::hip::par
#define  gputhrust_device_ptr thrust::device_ptr

 #define gpublasCheckErrors(fn) \
	 do { \
		 gpublasStatus_t __err = fn; \
		 if (__err != GPUBLAS_STATUS_SUCCESS) { \
			 fprintf(stderr, "Fatal hipblas error: %d (at %s:%d)\n", \
				 (int)(__err), \
				 __FILE__, __LINE__); \
			 fprintf(stderr, "*** FAILED - ABORTING\n"); \
			 exit(1); \
		 } \
	 } while(0);


 #define gpusolverCheckErrors(fn) \
	 do { \
		 gpusolverStatus_t __err = fn; \
		 if (__err != GPUSOLVER_STATUS_SUCCESS) { \
			 fprintf(stderr, "Fatal hipsolver error: %d (at %s:%d)\n", \
				 (int)(__err), \
				 __FILE__, __LINE__); \
			 fprintf(stderr, "*** FAILED - ABORTING\n"); \
			 exit(1); \
		 } \
	 } while(0);


#elif defined(HAVE_SYCL)

// theses oneDPL headers need to be set first
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "sycl_device.hpp"
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>

#define __host__
#define __global__ __inline__ __attribute__((always_inline))
#define __device__ __attribute__((always_inline))
static int gpuMemcpyHostToDevice{0};
static int gpuMemcpyDeviceToHost{0};
static int gpuMemcpyDeviceToDevice{0};
static int gpuHostMallocDefault{0};
using gpublasHandle_t = int;
using gpusolverDnHandle_t = int;
using gpuDoubleComplex = std::complex<double>;

#define threadIdx_x (item.get_local_id(2))
#define threadIdx_y (item.get_local_id(1))
#define blockIdx_x (item.get_group(2))
#define blockIdx_y (item.get_group(1))
#define blockIdx_z (item.get_group(0))
#define blockDim_x (item.get_local_range().get(2))
#define blockDim_y (item.get_local_range().get(1))
#define gridDim_x (item.get_group_range(2))
#define gridDim_y (item.get_group_range(1))
static inline double atomicAdd(double* addr, const double val) { return sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*addr).fetch_add( val ); }
//#define atomicAdd(addr,val) sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*addr).fetch_add( val )
#define atomicSub(addr,val) sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*addr).fetch_sub( val )
#define __threadfence() (sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device))

// the date 20240207 is the build date of oneapi/eng-compiler/2024.04.15.002
#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION > 20240227)
#define __syncthreads() (sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>()))
#define __syncwarp() (sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group()))
#else
#define __syncthreads() (sycl::group_barrier(sycl::ext::oneapi::experimental::this_group<3>()))
#define __syncwarp() (sycl::group_barrier(sycl::ext::oneapi::experimental::this_sub_group()))
#endif // SYCL_COMPILER_VERSION

static inline void gpuGetDeviceCount(int* count) { syclGetDeviceCount(count); }
static inline void gpuSetDevice(int deviceID) { syclSetDevice(deviceID); }
static inline void gpuGetDevice(int* deviceID) { syclGetDevice(deviceID); }
#define gpuDeviceReset() { }

static inline void gpuMemcpy(void* dst, const void* src, size_t count, int kind) {
  sycl_get_queue()->memcpy(dst, src, count).wait();
}
static inline void gpuDeviceSynchronize() { sycl_get_queue()->wait(); }
static inline void gpuMemcpyAsync(void* dst, const void* src, size_t count, int kind, sycl::queue* stream) {
  stream->memcpy(dst, src, count);
}

static inline void gpuMemset(void* ptr, int val, size_t size) {
  sycl_get_queue()->memset(ptr, val, size).wait();
}
static inline void gpuMemGetInfo(size_t* free, size_t* total) {
  *free = sycl_get_queue()->get_device().get_info<sycl::ext::intel::info::device::free_memory>();
  *total = sycl_get_queue()->get_device().get_info<sycl::info::device::global_mem_size>();
}

static inline void gpuMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, int kind, sycl::queue* stream) {
  stream->ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
}
static inline void gpuMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, int kind) {
  sycl_get_queue()->ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height).wait();
}

static inline void gpuMalloc(void** ptr, size_t size) {
  (*ptr) = (void*)sycl::malloc_device(size, *sycl_get_queue());
}
static inline void gpuHostMalloc(void** ptr, size_t size, int type) {
  (*ptr) = (void*)sycl::malloc_host(size, *sycl_get_queue());
}
static inline void gpuMallocHost(void** ptr, size_t size) {
  (*ptr) = (void*)sycl::malloc_host(size, *sycl_get_queue());
}
static inline void gpuFree(void* ptr) {
  sycl::free(ptr, sycl_get_queue()->get_context());
}
static inline void gpuStreamCreate(sycl::queue** syclStream) {
  (*syclStream) = new sycl::queue( sycl_get_queue()->get_context(), sycl_get_queue()->get_device(), asyncHandler, sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}} );
}
static inline void gpuStreamDestroy(sycl::queue* stream) {
  stream->wait();
  delete stream;
}
static inline void gpuStreamSynchronize(sycl::queue* stream) { stream->wait(); }
static inline void gpuFreeHost(void* ptr) {
  sycl::free(ptr, sycl_get_queue()->get_context());
  //::operator delete(ptr);
  //std::free(ptr);
}
static inline void gpuEventCreate(sycl::event** syclevent) { *syclevent = new sycl::event{}; }
static inline void gpuEventDestroy(sycl::event** event) { delete event; }
static inline void gpuEventRecord(sycl::event*& event, sycl::queue* stream) {
  *event = stream->ext_oneapi_submit_barrier();
}

static inline void gpuEventElapsedTime(float* ms, sycl::event* startEvent, sycl::event* endEvent) {
  *ms = (endEvent->get_profiling_info<sycl::info::event_profiling::command_end>() - startEvent->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
}

using gpuStream_t = sycl::queue*;
using gpuEvent_t = sycl::event*;
#define GPUBLAS_OP_N oneapi::mkl::transpose::nontrans
#define GPUBLAS_DIAG_UNIT oneapi::mkl::diag::unit
#define GPUBLAS_DIAG_NON_UNIT oneapi::mkl::diag::nonunit
#define GPUBLAS_SIDE_LEFT oneapi::mkl::side::left
#define GPUBLAS_SIDE_RIGHT oneapi::mkl::side::right
#define GPUBLAS_FILL_MODE_LOWER oneapi::mkl::uplo::lower
#define GPUBLAS_FILL_MODE_UPPER oneapi::mkl::uplo::upper

#define gputhrust oneapi::dpl
#define gputhrust_policy (oneapi::dpl::execution::make_device_policy(*(sycl_get_queue())))
#define gputhrust_device_ptr sycl::ext::intel::device_ptr

#define checkGPUErrors(fn) (fn);
#define checkGPUblas(fn) (fn);
#define checkGPU(fn) (fn);
static inline void gpuGetLastError() {}
#define gpublasCheckErrors(fn)						\
do {									\
        try {								\
                fn;                                                     \
        } catch (oneapi::mkl::exception const &ex) {                    \
                std::stringstream msg;                                  \
                msg << "Fatal oneMKL::BLAS error: " << __FILE__ << " : " << __LINE__ \
                    << std::endl;                                       \
                throw(std::runtime_error(ex.what()));                   \
                exit(1);                                                \
        }                                                               \
} while(0);

#define gpusolverCheckErrors(fn)                                        \
do {									\
        try {								\
                fn;                                                     \
        } catch (oneapi::mkl::lapack::exception const &ex) {            \
                std::stringstream msg;                                  \
                msg << "Fatal oneMKL::LAPACK error: " << __FILE__ << " : " << __LINE__ \
                    << std::endl;                                       \
                throw(std::runtime_error(ex.what()));                   \
                exit(1);                                                \
        }                                                               \
} while(0);

#endif // HAVE_CUDA

#endif /* __SUPERLU_GPUWRAPPER */
