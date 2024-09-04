#include <stdio.h>
#include <algorithm>
#include "gpu_wrapper.h"
#include "batch_block_copy.h"

#define BATCH_COPY_CPT	8
#define WARP_SIZE       32

__device__ __host__
inline int iDivUp( int a, int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template<class T>
__global__
void copy_block_vbatch_kernel(
	int* m_batch, int* n_batch, T** dest_ptrs, int* dest_ld_batch,
        T** src_ptrs, int* src_ld_batch
        #ifdef HAVE_SYCL
        , sycl::nd_item<3>& item
        #endif
)
{
  int batch_index = blockIdx_z;
  int row_index = blockDim_x * blockIdx_x + threadIdx_x;
  int col_index = (blockDim_y * blockIdx_y + threadIdx_y) * BATCH_COPY_CPT;
  int rows = m_batch[batch_index];
  int cols = n_batch[batch_index];

  if(row_index >= rows || col_index >= cols)
    return;

  int dest_ld = dest_ld_batch[batch_index];
  int src_ld = src_ld_batch[batch_index];

  T* dest_block = dest_ptrs[batch_index] + row_index + col_index * dest_ld;
  T* src_block = src_ptrs[batch_index] + row_index + col_index * src_ld;

  T reg_buffer[BATCH_COPY_CPT];

#pragma unroll
  for(int j = 0; j < BATCH_COPY_CPT; j++)
    if(j + col_index < cols)
      reg_buffer[j] = src_block[j * src_ld];

#pragma unroll
  for(int j = 0; j < BATCH_COPY_CPT; j++)
    if(j + col_index < cols)
      dest_block[j * dest_ld] = reg_buffer[j];
}

template<class T>
int copy_block_vbatch(
	gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
        T** dest_ptrs, int* dest_ld_batch, T** src_ptrs, int* src_ld_batch, int ops
)
{
  if(ops == 0 || max_m == 0 || max_n == 0)
    return 0;

  const int max_thread_y = 8;
  const int op_increment = 65535;

  int thread_x = WARP_SIZE, thread_y = std::min(max_thread_y, iDivUp(max_n, BATCH_COPY_CPT));
  int grid_x = iDivUp(max_m, thread_x), grid_y = iDivUp(max_n, thread_y * BATCH_COPY_CPT);

  for(int op_start = 0; op_start < ops; op_start += op_increment)
    {
      int batch_size = std::min(op_increment, ops - op_start);

      #ifdef HAVE_SYCL
      sycl::range<3> dimBlock(1, thread_y, thread_x);
      sycl::range<3> dimGrid(batch_size, grid_y, grid_x);
      stream->parallel_for(sycl::nd_range<3>(dimBlock * dimGrid, dimBlock), [=](auto item) {
        copy_block_vbatch_kernel<T>(m_batch, n_batch, dest_ptrs, dest_ld_batch, src_ptrs, src_ld_batch, item);
      });
      #else
      dim3 dimBlock(thread_x, thread_y, 1);
      dim3 dimGrid(grid_x, grid_y, batch_size);
      copy_block_vbatch_kernel<T><<< dimGrid, dimBlock, 0, stream >>>
        (m_batch, n_batch, dest_ptrs, dest_ld_batch, src_ptrs, src_ld_batch);
      #endif

      dest_ptrs += batch_size;
      src_ptrs += batch_size;
      dest_ld_batch += batch_size;
      src_ld_batch += batch_size;
      m_batch += batch_size;
      n_batch += batch_size;
    }

  #ifndef HAVE_SYCL
  if( cudaGetLastError() != cudaSuccess)
    return -1;
  #endif
  return 0;
}

// Variable batch interface
extern "C" int scopyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    float** dest_ptrs, int* dest_ld_batch, float** src_ptrs, int* src_ld_batch,
    int ops
)
{
	return copy_block_vbatch<float>(
		stream, m_batch, n_batch, max_m, max_n, dest_ptrs, dest_ld_batch,
        src_ptrs, src_ld_batch, ops
	);
}

extern "C" int dcopyBlock_vbatch(
    gpuStream_t stream, int* m_batch, int* n_batch, int max_m, int max_n,
    double** dest_ptrs, int* dest_ld_batch, double** src_ptrs, int* src_ld_batch,
    int ops
)
{
	return copy_block_vbatch<double>(
		stream, m_batch, n_batch, max_m, max_n, dest_ptrs, dest_ld_batch,
        src_ptrs, src_ld_batch, ops
	);
}
