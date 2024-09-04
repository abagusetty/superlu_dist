#include "batch_factorize.h"
#include <cstdio>

#ifdef HAVE_SYCL
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <functional>
#else
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/extrema.h>
#endif

#include <vector>
#include <iostream>

#include "lupanels.hpp" // For checkGPU - maybe move that function to utils
#include "gpuCommon.hpp"
#include "batch_factorize_marshall.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device functions and kernels 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ int_t find_entry_index_flat(int_t *index_list, int_t index, int_t n
                                       #ifdef HAVE_SYCL
                                       , sycl::nd_item<3> item
                                       #endif
                                       )
{
    int threadId = threadIdx_x;
    #ifdef HAVE_SYCL
    auto wrk_grp = item.get_group();
    int_t& idx = *sycl::ext::oneapi::group_local_memory_for_overwrite<int_t>(wrk_grp);    
    #else
    __shared__ int_t idx;
    #endif
    
    if (!threadId)    
        idx = -1;

    __syncthreads();

    int nThreads = blockDim_x;
    int blocksPerThreads = CEILING(n, nThreads);

    for (int_t blk = blocksPerThreads * threadIdx_x;
         blk < blocksPerThreads * (threadIdx_x + 1);
         blk++)
    {
        if (blk < n)
        {
            if(index == index_list[blk])
                idx = blk;
        }
    }
    __syncthreads();
    return idx;
}

__device__ int computeIndirectMapGPU_flat(int_t *rcS2D, int_t srcLen, int_t *srcVec, int_t src_first_index,
                                          int_t dstLen, int_t *dstVec, int_t dst_first_index, int_t *dstIdx
                                          #ifdef HAVE_SYCL
                                          , sycl::nd_item<3> item
                                          #endif
                                          )
{
    int threadId = threadIdx_x;
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        if (threadId < srcLen)
            rcS2D[threadId] = srcVec[threadId] - src_first_index;
        __syncthreads();
        return 0;
    }

    if (threadId < dstLen)
        dstIdx[dstVec[threadId] - dst_first_index] = threadId;
    __syncthreads();

    if (threadId < srcLen)
        rcS2D[threadId] = dstIdx[srcVec[threadId] - src_first_index];
    __syncthreads();

    return 0;
}

__global__ void scatterGPU_batch_flat(
    int_t k_st, int_t maxSuperSize, double **gemmBuff_ptrs, BatchDim_t *LDgemmBuff_batch,
    double **Unzval_br_new_ptr, int_t** Ucolind_br_ptr, double** Lnzval_bc_ptr, 
    int_t** Lrowind_bc_ptr, int_t** lblock_gid_ptrs, int_t **lblock_start_ptrs, 
    int_t *dperm_c_supno, int_t *xsup
    #ifdef HAVE_SYCL
    , sycl::nd_item<3> item,
    int_t* baseSharedPtr
    #endif
)
{
    int batch_index = blockIdx_z;
    int_t k = dperm_c_supno[k_st + batch_index];
    
    double* gemmBuff = gemmBuff_ptrs[batch_index];
    int_t *Ucolind_br = Ucolind_br_ptr[k];
    int_t *Lrowind_bc = Lrowind_bc_ptr[k];
    int_t *lblock_gid = lblock_gid_ptrs[k];
    int_t *lblock_start = lblock_start_ptrs[k];

    if(!Ucolind_br || !Lrowind_bc || !gemmBuff || !lblock_gid || !lblock_start)
        return;

    int_t L_blocks = Lrowind_bc[0];    
    int_t U_blocks = Ucolind_br[0];
    BatchDim_t LDgemmBuff = LDgemmBuff_batch[batch_index];

    int_t ii = 1 + blockIdx_x;
    int_t jj = blockIdx_y;

    if(ii >= L_blocks || jj >= U_blocks)
        return;

    // calculate gi, gj
    int threadId = threadIdx_x;

    int_t gi = lblock_gid[ii];
    int_t gj = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + jj];
    
    double *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    int_t dst_row_first_index, dst_col_first_index;
    int_t li = 0, lj = 0;

    if (gj > gi) // its in upanel
    {
        int_t* U_index_i = Ucolind_br_ptr[gi];
        int_t nub = U_index_i[0];
        lddst = U_index_i[2];
        lj = find_entry_index_flat(U_index_i + UB_DESCRIPTOR_NEWUCPP, gj, nub
                                   #ifdef HAVE_SYCL
                                   , item
                                   #endif
                                   );
        li = gi;
        int_t col_offset = U_index_i[UB_DESCRIPTOR_NEWUCPP + nub + lj];
        Dst = Unzval_br_new_ptr[gi] + lddst * col_offset;
        dstRowLen = lddst;
        dstRowList = NULL;
        dst_row_first_index = 0;
        dstColLen = U_index_i[UB_DESCRIPTOR_NEWUCPP + nub + lj + 1] - col_offset;
        dstColList = U_index_i + UB_DESCRIPTOR_NEWUCPP + 2 * nub + 1 + col_offset;
        dst_col_first_index = xsup[gj];
    }
    else
    {
        int_t* L_index_j = Lrowind_bc_ptr[gj], *ljblock_start = lblock_start_ptrs[gj];
        int_t nlb = L_index_j[0];
        lddst = L_index_j[1];
        li = find_entry_index_flat(lblock_gid_ptrs[gj], gi, nlb
                                   #ifdef HAVE_SYCL
                                   , item
                                   #endif                                   
                                   );
        lj = gj;
        int_t row_offset = ljblock_start[li];
        Dst = Lnzval_bc_ptr[gj] + row_offset;
        dstRowLen = ljblock_start[li + 1] - row_offset;
        dstRowList = L_index_j + BC_HEADER + (li + 1) * LB_DESCRIPTOR + row_offset;
        dst_row_first_index = xsup[gi];
        dstColLen = SuperSize(gj);
        dstColList = NULL;
        dst_col_first_index = 0;
    }

    // compute source row to dest row mapping
    #ifndef HAVE_SYCL
    extern __shared__ int_t baseSharedPtr[];
    #endif
    int_t *rowS2D = baseSharedPtr;
    int_t *colS2D = &rowS2D[maxSuperSize];
    int_t *dstIdx = &colS2D[maxSuperSize];

    int_t ublock_start = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + U_blocks + jj];
    int_t nrows = lblock_start[ii + 1] - lblock_start[ii];
    int_t ncols = Ucolind_br[UB_DESCRIPTOR_NEWUCPP + U_blocks + jj + 1] - ublock_start;

    int_t *lpanel_row_list = Lrowind_bc + BC_HEADER + (ii + 1) * LB_DESCRIPTOR + lblock_start[ii];
    int_t *upanel_col_list = Ucolind_br + UB_DESCRIPTOR_NEWUCPP + 2 * U_blocks + 1 + ublock_start;
    int_t lpanel_first_index = xsup[gi];
    int_t upanel_first_index = xsup[gj];

    computeIndirectMapGPU_flat(rowS2D, nrows, lpanel_row_list, lpanel_first_index,
                               dstRowLen, dstRowList, dst_row_first_index, dstIdx
                               #ifdef HAVE_SYCL
                               , item
                               #endif
                               );

    // compute source col to dest col mapping
    computeIndirectMapGPU_flat(colS2D, ncols, upanel_col_list, upanel_first_index,
                               dstColLen, dstColList, dst_col_first_index, dstIdx
                               #ifdef HAVE_SYCL
                               , item
                               #endif                               
                               );

    int nThreads = blockDim_x;
    int colsPerThreadBlock = nThreads / nrows;

    int_t rowOff = lblock_start[ii] - lblock_start[1];
    int_t colOff = ublock_start;

    double *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];
    int_t ldsrc = LDgemmBuff;

    // TODO: this seems inefficient
    if (threadId < nrows * colsPerThreadBlock)
    {
        /* 1D threads are logically arranged in 2D shape. */
        int i = threadId % nrows;
        int j = threadId / nrows;

#pragma unroll 4
        while (j < ncols)
        {

#define ATOMIC_SCATTER
// Atomic Scatter is need if I want to perform multiple Schur Complement
//  update concurrently
#ifdef ATOMIC_SCATTER
          atomicAdd(&Dst[rowS2D[i] + lddst * colS2D[j]], -Src[i + ldsrc * j]);
#else
          Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
#endif
          j += colsPerThreadBlock;
        }
    }

    __syncthreads();
}

inline void scatterGPU_batchDriver_flat(
    int_t k_st, int_t maxSuperSize, double **gemmBuff_ptrs, BatchDim_t *LDgemmBuff_batch,
    double **Unzval_br_new_ptr, int_t** Ucolind_br_ptr, double** Lnzval_bc_ptr, 
    int_t** Lrowind_bc_ptr, int_t** lblock_gid_ptrs, int_t **lblock_start_ptrs, 
    int_t *dperm_c_supno, int_t *xsup, int_t ldt, BatchDim_t max_ilen, BatchDim_t max_jlen, 
    BatchDim_t batchCount, gpuStream_t cuStream
)
{
    const BatchDim_t op_increment = 65535;
    
    for(BatchDim_t op_start = 0; op_start < batchCount; op_start += op_increment)
      {
        BatchDim_t batch_size = std::min(op_increment, batchCount - op_start);

        #ifdef HAVE_SYCL
        sycl::range<3> dimBlock(1, 1, ldt); // 1d thread
        sycl::range<3> dimGrid(batch_size, max_jlen, max_ilen);

        cuStream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int_t, 1> localmem(sycl::range<1>(3 * maxSuperSize), cgh);

          cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](auto item) {
            scatterGPU_batch_flat(k_st + op_start, maxSuperSize, gemmBuff_ptrs, LDgemmBuff_batch, Unzval_br_new_ptr,
                                  Ucolind_br_ptr, Lnzval_bc_ptr, Lrowind_bc_ptr, lblock_gid_ptrs, lblock_start_ptrs, 
                                  dperm_c_supno, xsup, item,
                                  localmem.get_multi_ptr<sycl::access::decorated::yes>().get());
          }); });
        #else
        dim3 dimBlock(ldt); // 1d thread
        dim3 dimGrid(max_ilen, max_jlen, batch_size);
        size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

        scatterGPU_batch_flat<<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            k_st + op_start, maxSuperSize, gemmBuff_ptrs, LDgemmBuff_batch, Unzval_br_new_ptr,
            Ucolind_br_ptr, Lnzval_bc_ptr, Lrowind_bc_ptr, lblock_gid_ptrs, lblock_start_ptrs, 
            dperm_c_supno, xsup);
        #endif
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshalling workspace
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BatchLUMarshallData::BatchLUMarshallData()
{
    dev_diag_ptrs = dev_panel_ptrs = NULL;
    dev_diag_ld_array = dev_diag_dim_array = dev_info_array = NULL;
    dev_panel_ld_array = dev_panel_dim_array = NULL;
}

BatchLUMarshallData::~BatchLUMarshallData()
{
    checkGPU(gpuFree(dev_diag_ptrs));
    checkGPU(gpuFree(dev_panel_ptrs));
    checkGPU(gpuFree(dev_diag_ld_array));
    checkGPU(gpuFree(dev_diag_dim_array));
    checkGPU(gpuFree(dev_info_array));
    checkGPU(gpuFree(dev_panel_ld_array));
    checkGPU(gpuFree(dev_panel_dim_array));
}

void BatchLUMarshallData::setBatchSize(BatchDim_t batch_size)
{
    checkGPU(gpuMalloc((void**)&dev_diag_ptrs, batch_size * sizeof(double*)));
    checkGPU(gpuMalloc((void**)&dev_panel_ptrs, batch_size * sizeof(double*)));

    checkGPU(gpuMalloc((void**)&dev_diag_ld_array, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_diag_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_info_array, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_panel_ld_array, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_panel_dim_array, (batch_size + 1) * sizeof(BatchDim_t)));
}

BatchSCUMarshallData::BatchSCUMarshallData()
{
    dev_A_ptrs = dev_B_ptrs = dev_C_ptrs = NULL;
    dev_lda_array = dev_ldb_array = dev_ldc_array = NULL;
    dev_m_array = dev_n_array = dev_k_array = NULL;
    dev_ist = dev_iend = dev_jst = dev_jend = NULL;
}

BatchSCUMarshallData::~BatchSCUMarshallData()
{
    checkGPU(gpuFree(dev_A_ptrs));
    checkGPU(gpuFree(dev_B_ptrs));
    checkGPU(gpuFree(dev_C_ptrs));
    checkGPU(gpuFree(dev_lda_array));
    checkGPU(gpuFree(dev_ldb_array));
    checkGPU(gpuFree(dev_ldc_array));
    checkGPU(gpuFree(dev_m_array));
    checkGPU(gpuFree(dev_n_array));
    checkGPU(gpuFree(dev_k_array));
    checkGPU(gpuFree(dev_ist));
    checkGPU(gpuFree(dev_iend));
    checkGPU(gpuFree(dev_jst));
    checkGPU(gpuFree(dev_jend));
}

void BatchSCUMarshallData::setBatchSize(BatchDim_t batch_size)
{
    checkGPU(gpuMalloc((void**)&dev_A_ptrs, batch_size * sizeof(double*)));
    checkGPU(gpuMalloc((void**)&dev_B_ptrs, batch_size * sizeof(double*)));
    checkGPU(gpuMalloc((void**)&dev_C_ptrs, batch_size * sizeof(double*)));

    checkGPU(gpuMalloc((void**)&dev_lda_array, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_ldb_array, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_ldc_array, batch_size * sizeof(BatchDim_t)));

    checkGPU(gpuMalloc((void**)&dev_m_array, (batch_size + 1) * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_n_array, (batch_size + 1) * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_k_array, (batch_size + 1) * sizeof(BatchDim_t)));

    checkGPU(gpuMalloc((void**)&dev_ist, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_iend, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_jst, batch_size * sizeof(BatchDim_t)));
    checkGPU(gpuMalloc((void**)&dev_jend, batch_size * sizeof(BatchDim_t)));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Marshalling routines for batched execution 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void marshallBatchedLUData(BatchFactorizeWorkspace* ws, int_t k_st, int_t k_end)
{
    BatchLUMarshallData& mdata = ws->marshall_data;
    dLocalLU_t& d_localLU = ws->d_localLU;

    mdata.batchsize = k_end - k_st;

    MarshallLUFunc_flat func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    gputhrust::for_each(
        gputhrust_policy, gputhrust::counting_iterator<int_t>(0),
        gputhrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

void marshallBatchedTRSMUData(BatchFactorizeWorkspace* ws, int_t k_st, int_t k_end)
{
    BatchLUMarshallData& mdata = ws->marshall_data;
    dLocalLU_t& d_localLU = ws->d_localLU;

    mdata.batchsize = k_end - k_st;

    MarshallTRSMUFunc_flat func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, 
        d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, d_localLU.Lnzval_bc_ptr, 
        d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    gputhrust::for_each(
        gputhrust_policy, gputhrust::counting_iterator<int_t>(0),
        gputhrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

void marshallBatchedTRSMLData(BatchFactorizeWorkspace* ws, int_t k_st, int_t k_end)
{
    BatchLUMarshallData& mdata = ws->marshall_data;
    dLocalLU_t& d_localLU = ws->d_localLU;
    
    mdata.batchsize = k_end - k_st;

    MarshallTRSMLFunc_flat func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array,
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup
    );

    gputhrust::for_each(
        gputhrust_policy, gputhrust::counting_iterator<int_t>(0),
        gputhrust::counting_iterator<int_t>(mdata.batchsize), func
    );
}

void marshallBatchedSCUData(BatchFactorizeWorkspace* ws, int_t k_st, int_t k_end)
{
    BatchSCUMarshallData& sc_mdata = ws->sc_marshall_data;
    dLocalLU_t& d_localLU = ws->d_localLU;

    sc_mdata.batchsize = k_end - k_st;
    
    gputhrust::counting_iterator<int_t> start(0), end(sc_mdata.batchsize);
    
    MarshallSCUFunc_flat func(
        k_st, sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array, 
        sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array,
        sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend, d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, 
        d_localLU.Lnzval_bc_ptr, d_localLU.Lrowind_bc_ptr, ws->perm_c_supno, ws->xsup, ws->gemm_buff_ptrs
    );

    gputhrust::for_each(gputhrust_policy, start, end, func);

    // Set the max dims in the marshalled data 
    sc_mdata.max_m = gputhrust::reduce(gputhrust_policy, sc_mdata.dev_m_array, sc_mdata.dev_m_array + sc_mdata.batchsize, 0, gputhrust::maximum<BatchDim_t>());
    sc_mdata.max_n = gputhrust::reduce(gputhrust_policy, sc_mdata.dev_n_array, sc_mdata.dev_n_array + sc_mdata.batchsize, 0, gputhrust::maximum<BatchDim_t>());
    sc_mdata.max_k = gputhrust::reduce(gputhrust_policy, sc_mdata.dev_k_array, sc_mdata.dev_k_array + sc_mdata.batchsize, 0, gputhrust::maximum<BatchDim_t>());
    // sc_mdata.max_ilen = gputhrust::transform_reduce(gputhrust_policy, start, end, element_diff<BatchDim_t>(sc_mdata.dev_ist, sc_mdata.dev_iend), 0, gputhrust::maximum<BatchDim_t>());
    // sc_mdata.max_jlen = gputhrust::transform_reduce(gputhrust_policy, start, end, element_diff<BatchDim_t>(sc_mdata.dev_jst, sc_mdata.dev_jend), 0, gputhrust::maximum<BatchDim_t>());

    sc_mdata.max_ilen = gputhrust::transform_reduce(gputhrust_policy, start, end, 0, gputhrust::maximum<BatchDim_t>(), element_diff<BatchDim_t>(sc_mdata.dev_ist, sc_mdata.dev_iend));
    sc_mdata.max_jlen = gputhrust::transform_reduce(gputhrust_policy, start, end, 0, gputhrust::maximum<BatchDim_t>(), element_diff<BatchDim_t>(sc_mdata.dev_jst, sc_mdata.dev_jend));
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility routines
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct BatchLDataSizeAssign_Func {
    int_t** Lrowind_bc_ptr;
    int64_t *d_lblock_gid_offsets, *d_lblock_start_offsets;

    BatchLDataSizeAssign_Func(int_t** Lrowind_bc_ptr, int64_t* d_lblock_gid_offsets, int64_t* d_lblock_start_offsets)
    {
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->d_lblock_gid_offsets = d_lblock_gid_offsets;
        this->d_lblock_start_offsets = d_lblock_start_offsets;
    }

    __device__ void operator()(const int_t &i) const
    {   
        if(i == 0)
            d_lblock_gid_offsets[i] = d_lblock_start_offsets[i] = 0;
        else
        {
            int_t *Lrowind_bc = Lrowind_bc_ptr[i - 1];
            d_lblock_gid_offsets[i] = (Lrowind_bc ? Lrowind_bc[0] : 0);
            d_lblock_start_offsets[i] = (Lrowind_bc ? Lrowind_bc[0] + 1 : 0);
        }
    }
};

struct BatchLDataAssign_Func {
    int_t **Lrowind_bc_ptr, **d_lblock_gid_ptrs, **d_lblock_start_ptrs;

    BatchLDataAssign_Func(int_t** Lrowind_bc_ptr, int_t** d_lblock_gid_ptrs, int_t** d_lblock_start_ptrs)
    {
        this->Lrowind_bc_ptr = Lrowind_bc_ptr;
        this->d_lblock_gid_ptrs = d_lblock_gid_ptrs;
        this->d_lblock_start_ptrs = d_lblock_start_ptrs;
    }

    __device__ void operator()(const int_t &i) const
    {   
        int_t *Lrowind_bc = Lrowind_bc_ptr[i];
        if(!Lrowind_bc)
            d_lblock_gid_ptrs[i] = d_lblock_start_ptrs[i] = NULL;
        else
        {   
            int_t *block_gids = d_lblock_gid_ptrs[i], *block_starts = d_lblock_start_ptrs[i];
            int_t nblocks = Lrowind_bc[0], Lptr = BC_HEADER, psum = 0;
            for(int_t b = 0; b < nblocks; b++)
            {
                block_gids[b] = Lrowind_bc[Lptr];
                int_t nrows = Lrowind_bc[Lptr + 1];
                block_starts[b] = psum;
                psum += nrows;
                Lptr += nrows + LB_DESCRIPTOR;
            }
            block_starts[nblocks] = psum;
        }
    }
};

void computeLBlockData(BatchFactorizeWorkspace* ws, int_t nsupers)
{
    dLocalLU_t& d_localLU = ws->d_localLU;

    // Allocate memory for the offsets and the pointers 
    checkGPU( gpuMalloc((void**)&ws->d_lblock_gid_offsets, sizeof(int64_t) * (nsupers + 1)) );
    checkGPU( gpuMalloc((void**)&ws->d_lblock_start_offsets, sizeof(int64_t) * (nsupers + 1)) );
    checkGPU( gpuMalloc((void**)&ws->d_lblock_gid_ptrs, sizeof(int_t*) * nsupers) );
    checkGPU( gpuMalloc((void**)&ws->d_lblock_start_ptrs, sizeof(int_t*) * nsupers) );

    // Initialize to the block counts for each panel
    gputhrust::for_each(
        gputhrust_policy, gputhrust::counting_iterator<int_t>(0), 
        gputhrust::counting_iterator<int_t>(nsupers + 1), BatchLDataSizeAssign_Func(
            d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_offsets, ws->d_lblock_start_offsets
    ) );
    
    // Do an inclusive scan to compute offsets and get the total amount of blocks 
    ws->total_l_blocks = *(gputhrust_device_ptr<int64_t>(
        gputhrust::inclusive_scan(
            gputhrust_policy, ws->d_lblock_gid_offsets + 1, 
            ws->d_lblock_gid_offsets + nsupers + 1, ws->d_lblock_gid_offsets + 1
    ) ) - 1);

    ws->total_start_size = *(gputhrust_device_ptr<int64_t>(
        gputhrust::inclusive_scan(
            gputhrust_policy, ws->d_lblock_start_offsets + 1, 
            ws->d_lblock_start_offsets + nsupers + 1, ws->d_lblock_start_offsets + 1
    ) ) - 1);

    // Allocate the block data 
    checkGPU( gpuMalloc((void**)&ws->d_lblock_gid_dat, sizeof(int_t) * ws->total_l_blocks) );
    checkGPU( gpuMalloc((void**)&ws->d_lblock_start_dat, sizeof(int_t) * ws->total_start_size) );

    // Generate the pointers 
    generateOffsetPointers(ws->d_lblock_gid_dat, ws->d_lblock_gid_offsets, ws->d_lblock_gid_ptrs, nsupers);
    generateOffsetPointers(ws->d_lblock_start_dat, ws->d_lblock_start_offsets, ws->d_lblock_start_ptrs, nsupers);

    // Now copy the data over from d_localLU
    gputhrust::for_each(
        gputhrust_policy, gputhrust::counting_iterator<int_t>(0), 
        gputhrust::counting_iterator<int_t>(nsupers), BatchLDataAssign_Func(
        d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_ptrs, ws->d_lblock_start_ptrs
    ) );
}

void batchAllocateGemmBuffers(
    BatchFactorizeWorkspace* ws, dLUstruct_t *LUstruct, dtrf3Dpartition_t *trf3Dpartition, 
    gridinfo3d_t *grid3d
)
{
    int_t mxLeafNode = trf3Dpartition->mxLeafNode;

    // TODO: is this necessary if this is being done on a single node?
    int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

    std::vector<int64_t> gemmCsizes(mxLeafNode, 0);
	int_t mx_fsize = 0;
	
	for (int_t ilvl = 0; ilvl < maxLvl; ++ilvl) 
    {
	    int_t treeId = trf3Dpartition->myTreeIdxs[ilvl];
	    sForest_t* sforest = trf3Dpartition->sForests[treeId];
	    if (sforest)
        {
            int_t *perm_c_supno = sforest->nodeList;
            mx_fsize = std::max(mx_fsize, sforest->nNodes);

            int_t maxTopoLevel = sforest->topoInfo.numLvl;
            for (int_t topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl) 
            {
                int_t k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
                int_t k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];
            
                for (int_t k0 = k_st; k0 < k_end; ++k0) 
                {
                    int_t offset = k0 - k_st;
                    int_t k = perm_c_supno[k0];
                    int_t* L_data = LUstruct->Llu->Lrowind_bc_ptr[k];
                    int_t* U_data = LUstruct->Llu->Ucolind_br_ptr[k];
                    if(L_data && U_data)
                    {
                        int_t Csize = L_data[1] * U_data[1];
                        gemmCsizes[offset] = SUPERLU_MAX(gemmCsizes[offset], Csize);
                    }
                }
		    }
	    }
	}
    // Allocate the gemm buffers 
    checkGPU( gpuMalloc((void**)&(ws->gemm_buff_ptrs), sizeof(double*) * mxLeafNode) );
    checkGPU( gpuMalloc((void**)&(ws->gemm_buff_offsets), sizeof(int64_t) * (mxLeafNode + 1)) );

    // Copy the host offset to the device 
    checkGPU( gpuMemcpy( ws->gemm_buff_offsets + 1, gemmCsizes.data(), mxLeafNode * sizeof(int64_t), gpuMemcpyHostToDevice) );
    *(gputhrust_device_ptr<int64_t>(ws->gemm_buff_offsets)) = 0;

    int64_t total_entries = *(gputhrust_device_ptr<int64_t>(
        gputhrust::inclusive_scan(
            gputhrust_policy, ws->gemm_buff_offsets + 1, 
            ws->gemm_buff_offsets + mxLeafNode + 1, ws->gemm_buff_offsets + 1
    ) ) - 1);

    // Allocate the base memory and generate the pointers on the device
    checkGPU(gpuMalloc((void**)&(ws->gemm_buff_base), sizeof(double) * total_entries));
    generateOffsetPointers(ws->gemm_buff_base, ws->gemm_buff_offsets, ws->gemm_buff_ptrs, mxLeafNode);

    // Allocate GPU copy for the node list 
    checkGPU(gpuMalloc((void**)&(ws->perm_c_supno), sizeof(int_t) * mx_fsize));
}

void copyHostLUDataToGPU(BatchFactorizeWorkspace* ws, dLocalLU_t* host_Llu, int_t nsupers)
{
    dLocalLU_t& d_localLU = ws->d_localLU;

    // Allocate data, offset and ptr arrays for the indices and lower triangular blocks 
    d_localLU.Lrowind_bc_cnt = host_Llu->Lrowind_bc_cnt;
    checkGPU( gpuMalloc((void**)&(d_localLU.Lrowind_bc_dat), d_localLU.Lrowind_bc_cnt * sizeof(int_t)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Lrowind_bc_offset), nsupers * sizeof(long int)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Lrowind_bc_ptr), nsupers * sizeof(int_t*)) );

    d_localLU.Lnzval_bc_cnt = host_Llu->Lnzval_bc_cnt;
    checkGPU( gpuMalloc((void**)&(d_localLU.Lnzval_bc_dat), d_localLU.Lnzval_bc_cnt * sizeof(double)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Lnzval_bc_offset), nsupers * sizeof(long int)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Lnzval_bc_ptr), nsupers * sizeof(double*)) );

    // Allocate data, offset and ptr arrays for the indices and upper triangular blocks 
    d_localLU.Ucolind_br_cnt = host_Llu->Ucolind_br_cnt;
    checkGPU( gpuMalloc((void**)&(d_localLU.Ucolind_br_dat), d_localLU.Ucolind_br_cnt * sizeof(int_t)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Ucolind_br_offset), nsupers * sizeof(int64_t)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Ucolind_br_ptr), nsupers * sizeof(int_t*)) );

    d_localLU.Unzval_br_new_cnt = host_Llu->Unzval_br_new_cnt;
    checkGPU( gpuMalloc((void**)&(d_localLU.Unzval_br_new_dat), d_localLU.Unzval_br_new_cnt * sizeof(double)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Unzval_br_new_offset), nsupers * sizeof(int64_t)) );
    checkGPU( gpuMalloc((void**)&(d_localLU.Unzval_br_new_ptr), nsupers * sizeof(double*)) );

    // Copy the index and nzval data over to the GPU 
    checkGPU( gpuMemcpy(d_localLU.Lrowind_bc_dat, host_Llu->Lrowind_bc_dat, d_localLU.Lrowind_bc_cnt * sizeof(int_t), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Lrowind_bc_offset, host_Llu->Lrowind_bc_offset, nsupers * sizeof(long int), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Lnzval_bc_dat, host_Llu->Lnzval_bc_dat, d_localLU.Lnzval_bc_cnt * sizeof(double), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Lnzval_bc_offset, host_Llu->Lnzval_bc_offset, nsupers * sizeof(long int), gpuMemcpyHostToDevice) );
    
    checkGPU( gpuMemcpy(d_localLU.Ucolind_br_dat, host_Llu->Ucolind_br_dat, d_localLU.Ucolind_br_cnt * sizeof(int_t), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Ucolind_br_offset, host_Llu->Ucolind_br_offset, nsupers * sizeof(int64_t), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Unzval_br_new_dat, host_Llu->Unzval_br_new_dat, d_localLU.Unzval_br_new_cnt * sizeof(double), gpuMemcpyHostToDevice) );
    checkGPU( gpuMemcpy(d_localLU.Unzval_br_new_offset, host_Llu->Unzval_br_new_offset, nsupers * sizeof(int64_t), gpuMemcpyHostToDevice) );
    
    // Generate the pointers using the offsets 
    generateOffsetPointers(d_localLU.Lrowind_bc_dat, d_localLU.Lrowind_bc_offset, d_localLU.Lrowind_bc_ptr, nsupers);
    generateOffsetPointers(d_localLU.Lnzval_bc_dat, d_localLU.Lnzval_bc_offset, d_localLU.Lnzval_bc_ptr, nsupers);
    generateOffsetPointers(d_localLU.Ucolind_br_dat, d_localLU.Ucolind_br_offset, d_localLU.Ucolind_br_ptr, nsupers);
    generateOffsetPointers(d_localLU.Unzval_br_new_dat, d_localLU.Unzval_br_new_offset, d_localLU.Unzval_br_new_ptr, nsupers);

    // Copy the L data for global ids and block offsets into a more parallel friendly data structure 
    computeLBlockData(ws, nsupers);
}

#ifdef HAVE_MAGMA
extern "C" void
magmablas_dtrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);
#endif

void dFactBatchSolve(BatchFactorizeWorkspace* ws, int_t k_st, int_t k_end)
{
#ifdef HAVE_MAGMA
    dLocalLU_t& d_localLU = ws->d_localLU;
    BatchLUMarshallData& mdata = ws->marshall_data;
    BatchSCUMarshallData& sc_mdata = ws->sc_marshall_data;

    // Diagonal block batched LU decomposition   
    marshallBatchedLUData(ws, k_st, k_end);
    
    int_t info = magma_dgetrf_vbatched(
        mdata.dev_diag_dim_array, mdata.dev_diag_dim_array, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        NULL, mdata.dev_info_array, mdata.batchsize, 
        ws->magma_queue
    );
    
    // Upper panel batched triangular solves
    marshallBatchedTRSMUData(ws, k_st, k_end);

    magmablas_dtrsm_vbatched_nocheck(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 
        mdata.dev_diag_dim_array, mdata.dev_panel_dim_array, 1.0, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, 
        mdata.batchsize, ws->magma_queue
    );

    // Lower panel batched triangular solves
    marshallBatchedTRSMLData(ws, k_st, k_end);

    magmablas_dtrsm_vbatched_nocheck(
        MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
        mdata.dev_panel_dim_array, mdata.dev_diag_dim_array, 1.0, 
        mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, 
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, 
        mdata.batchsize, ws->magma_queue
    );

    // Batched schur complement updates 
    marshallBatchedSCUData(ws, k_st, k_end);
    
    magmablas_dgemm_vbatched_max_nocheck (
        MagmaNoTrans, MagmaNoTrans, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array,
        1.0, sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array,
        0.0, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array, sc_mdata.batchsize,
        sc_mdata.max_m, sc_mdata.max_n, sc_mdata.max_k, ws->magma_queue
    );
    
    scatterGPU_batchDriver_flat(
        k_st, ws->maxSuperSize, sc_mdata.dev_C_ptrs, sc_mdata.dev_ldc_array,
        d_localLU.Unzval_br_new_ptr, d_localLU.Ucolind_br_ptr, d_localLU.Lnzval_bc_ptr, 
        d_localLU.Lrowind_bc_ptr, ws->d_lblock_gid_ptrs, ws->d_lblock_start_ptrs, 
        ws->perm_c_supno, ws->xsup, ws->ldt, sc_mdata.max_ilen, sc_mdata.max_jlen, 
        sc_mdata.batchsize, ws->stream
    );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main factorization routiunes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" {
int dsparseTreeFactorBatchGPU(BatchFactorizeWorkspace* ws, sForest_t *sforest)
{
    int_t nnodes = sforest->nNodes; 

    if(nnodes < 1)
        return 1;
    
    // Host list of nodes in the order of factorization copied to the GPU 
    int_t *perm_c_supno = sforest->nodeList; 
    checkGPU(gpuMemcpy(ws->perm_c_supno, perm_c_supno, sizeof(int_t) * nnodes, gpuMemcpyHostToDevice));

    // Tree containing the supernode limits per level 
    treeTopoInfo_t *treeTopoInfo = &sforest->topoInfo;
    int_t maxTopoLevel = treeTopoInfo->numLvl;
    int_t *eTreeTopLims = treeTopoInfo->eTreeTopLims;

    for(int_t topoLvl = 0; topoLvl < maxTopoLevel; topoLvl++)
        dFactBatchSolve(ws, eTreeTopLims[topoLvl], eTreeTopLims[topoLvl + 1]);

    return 0;
}

BatchFactorizeWorkspace* getBatchFactorizeWorkspace(
    int_t nsupers, int_t ldt, dtrf3Dpartition_t *trf3Dpartition, dLUstruct_t *LUstruct, 
    gridinfo3d_t *grid3d, superlu_dist_options_t *options, SuperLUStat_t *stat, int *info
)
{
#ifdef HAVE_MAGMA
    BatchFactorizeWorkspace* ws = new BatchFactorizeWorkspace();
    
    // TODO: this should be assigned somewhere 
    const int device_id = 0;

    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t n = xsup[nsupers];
    gridinfo_t *grid = &(grid3d->grid2d);

    double tic = SuperLU_timer_();

    pdconvert_flatten_skyline2UROWDATA(options, grid, LUstruct, stat, n);

    double convert_time = SuperLU_timer_() - tic;

    // TODO: determine if ldt is supposed to be the same as maxSuperSize?
    ws->ldt = ws->maxSuperSize = ldt;
    ws->nsupers = nsupers;

    // Set up device handles 
    checkGPU( gpuStreamCreate(&ws->stream) );
    gpublasCreate( &ws->cuhandle );
    magma_queue_create_from_cuda(device_id, ws->stream, ws->cuhandle, NULL, &ws->magma_queue);

    // Copy the xsup to the GPU 
    tic = SuperLU_timer_();
    checkGPU(gpuMalloc((void**)&ws->xsup, (nsupers + 1) * sizeof(int_t)));
    checkGPU(gpuMemcpy(ws->xsup, xsup, (nsupers + 1) * sizeof(int_t), gpuMemcpyHostToDevice));

    // Copy the flattened LU data over to the GPU 
    // TODO: I currently have to make a GPU friendly copy of the globa ids of blocks within L
    // and compute block offsets. Can this be avoided with a change to the L index structure?
    copyHostLUDataToGPU(ws, LUstruct->Llu, nsupers);

    double copy_time = SuperLU_timer_() - tic;

    // Allocate marhsalling workspace
    tic = SuperLU_timer_();
    ws->marshall_data.setBatchSize(trf3Dpartition->mxLeafNode);
    ws->sc_marshall_data.setBatchSize(trf3Dpartition->mxLeafNode);

    // Determine buffer sizes for schur complement updates and supernode lists 
    batchAllocateGemmBuffers(ws, LUstruct, trf3Dpartition, grid3d);
    double ws_time = SuperLU_timer_() - tic;
    
    printf("\tSky2UROWDATA Convert time = %.4f\n", convert_time);
    printf("\tH2D Copy time = %.4f\n", copy_time);
    printf("\tWorkspace alloc time = %.4f\n", ws_time);

    return ws;
#endif
}

void copyGPULUDataToHost(
    BatchFactorizeWorkspace* ws, dLUstruct_t *LUstruct, gridinfo3d_t *grid3d,
    SCT_t *SCT_, superlu_dist_options_t *options, SuperLUStat_t *stat
)
{
    dLocalLU_t& d_localLU = ws->d_localLU;
    dLocalLU_t* host_Llu = LUstruct->Llu;

    double tic = SuperLU_timer_();
    
    // Only need to copy the nzval data arrays when moving from the GPU to the Host 
    checkGPU( gpuMemcpy(host_Llu->Lnzval_bc_dat, d_localLU.Lnzval_bc_dat, d_localLU.Lnzval_bc_cnt * sizeof(double), gpuMemcpyDeviceToHost) );
    checkGPU( gpuMemcpy(host_Llu->Unzval_br_new_dat, d_localLU.Unzval_br_new_dat, d_localLU.Unzval_br_new_cnt * sizeof(double), gpuMemcpyDeviceToHost) );
    
    double copy_time = SuperLU_timer_() - tic;

    // Convert the host data from block row to skyline 
    int_t* xsup = LUstruct->Glu_persist->xsup;
    int_t n = xsup[ws->nsupers];
    gridinfo_t *grid = &(grid3d->grid2d);

    tic = SuperLU_timer_();
    pdconvertUROWDATA2skyline(options, grid, LUstruct, stat, n);
    double convert_time = SuperLU_timer_() - tic;

    printf("\tD2H Copy time = %.4f\n", copy_time);
    printf("\tConvert time = %.4f\n", convert_time);
}

void freeBatchFactorizeWorkspace(BatchFactorizeWorkspace* ws)
{
    checkGPU( gpuFree(ws->d_lblock_gid_dat) );
    checkGPU( gpuFree(ws->d_lblock_gid_offsets) );
    checkGPU( gpuFree(ws->d_lblock_gid_ptrs) );
    checkGPU( gpuFree(ws->d_lblock_start_dat) );
    checkGPU( gpuFree(ws->d_lblock_start_offsets) );
    checkGPU( gpuFree(ws->d_lblock_start_ptrs) );
    checkGPU( gpuFree(ws->gemm_buff_base) );
    checkGPU( gpuFree(ws->gemm_buff_offsets) );
    checkGPU( gpuFree(ws->gemm_buff_ptrs) );
    checkGPU( gpuFree(ws->perm_c_supno) );
    checkGPU( gpuFree(ws->xsup) );

    dLocalLU_t& d_localLU = ws->d_localLU;
    checkGPU( gpuFree(d_localLU.Lrowind_bc_dat) );
    checkGPU( gpuFree(d_localLU.Lrowind_bc_offset) );
    checkGPU( gpuFree(d_localLU.Lrowind_bc_ptr) );
    checkGPU( gpuFree(d_localLU.Lnzval_bc_dat) );
    checkGPU( gpuFree(d_localLU.Lnzval_bc_offset) );
    checkGPU( gpuFree(d_localLU.Lnzval_bc_ptr) );
    checkGPU( gpuFree(d_localLU.Ucolind_br_dat) );
    checkGPU( gpuFree(d_localLU.Ucolind_br_offset) );
    checkGPU( gpuFree(d_localLU.Ucolind_br_ptr) );
    checkGPU( gpuFree(d_localLU.Unzval_br_new_dat) );
    checkGPU( gpuFree(d_localLU.Unzval_br_new_offset) );
    checkGPU( gpuFree(d_localLU.Unzval_br_new_ptr) );
#ifdef HAVE_MAGMA
    magma_queue_destroy(ws->magma_queue);
    gpublasDestroy( ws->cuhandle );    
#endif
    checkGPU( gpuStreamDestroy(ws->stream) );
}

}
