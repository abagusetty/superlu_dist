#pragma once

#include "gpu_wrapper.h"

#ifdef HAVE_SYCL
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <functional>
#else
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#endif

#include "lupanels_GPU.hpp"
#include "superlu_ddefs.h"
#include "lupanels.hpp"
#include "gpuCommon.hpp"
#ifndef HAVE_SYCL
#include "gpublas_gpusolver_wrappers.hpp"
#endif


#define USABLE_GPU_MEM_FRACTION 0.9

size_t getGPUMemPerProcs(MPI_Comm baseCommunicator);

template <typename Ftype>
__global__ void indirectCopy(Ftype *dest, Ftype *src, int_t *idx, int n
                             #ifdef HAVE_SYCL
                             , sycl::nd_item<3>& item
                             #endif
                             )
{
    #ifdef HAVE_SYCL
    int i = static_cast<int>(item.get_global_id(2));
    #else
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    #endif
    if (i < n)
      dest[idx[i]] = src[i];
}

/**
 * @file schurCompUpdate.cu
 * @brief This function copies the packed buffers to GPU and performs the sparse
   initialization on GPU call indirectCopy, this is the kernel
 * @param gpuValBasePtr is the base pointer of the GPU matrix
 * @param valBufferPacked is the packed buffer of the matrix
 * @param valIdx is the index of the packed buffer
 */
template <typename Ftype>
void copyToGPU(Ftype *gpuValBasePtr, std::vector<Ftype> &valBufferPacked,
               std::vector<int_t> &valIdx)
{
    int nnzCount = valBufferPacked.size();
    // calculate the size of the packed buffers
    int_t gpuLvalSizePacked = nnzCount * sizeof(Ftype);
    int_t gpuLidxSizePacked = nnzCount * sizeof(int_t);
    // allocate the memory for the packed buffers on GPU
    Ftype *dlvalPacked;
    int_t *dlidxPacked;
    checkGPU(gpuMalloc((void**)&dlvalPacked, gpuLvalSizePacked));
    checkGPU(gpuMalloc((void**)&dlidxPacked, gpuLidxSizePacked));
    // copy the packed buffers from CPU to GPU
    checkGPU(gpuMemcpy(dlvalPacked, valBufferPacked.data(), gpuLvalSizePacked, gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(dlidxPacked, valIdx.data(), gpuLidxSizePacked, gpuMemcpyHostToDevice));
    // perform the sparse initialization on GPU call indirectCopy
    const int ThreadblockSize = 256;
    int nThreadBlocks = (nnzCount + ThreadblockSize - 1) / ThreadblockSize;
    #ifdef HAVE_SYCL
    sycl::range<3> global(1, 1, nThreadBlocks * ThreadblockSize);
    sycl::range<3> local(1, 1, ThreadblockSize);
    sycl_get_queue()->parallel_for(sycl::nd_range<3>(global, local), [=](auto item) {
      indirectCopy(gpuValBasePtr, dlvalPacked, dlidxPacked, nnzCount, item);
    });
    #else
    indirectCopy<<<nThreadBlocks, ThreadblockSize>>>(
        gpuValBasePtr, dlvalPacked, dlidxPacked, nnzCount);
    #endif
    // wait for it to finish and free dlvalPacked and dlidxPacked
    checkGPU(gpuDeviceSynchronize());
    checkGPU(gpuFree(dlvalPacked));
    checkGPU(gpuFree(dlidxPacked));
}

// copy the panel to GPU
template <typename Ftype>
void copyToGPU_Sparse(Ftype *gpuValBasePtr, Ftype *valBuffer, int_t gpuLvalSize)
{
    // sparse Initialization for GPU, this is the experimental code
    // find non-zero elements in the panel, their location and values  and copy to GPU
    int numFtypes = gpuLvalSize / sizeof(Ftype);
    std::vector<Ftype> valBufferPacked;
    std::vector<int_t> valIdx;
    for (int_t i = 0; i < numFtypes; i++)
    {
        if (valBuffer[i] != 0)
        {
            valBufferPacked.push_back(valBuffer[i]);
            valIdx.push_back(i);
        }
    }
    printf("%d non-zero elements in the panel, wrt original=%d\n", valBufferPacked.size(), numFtypes);
    // get the size of the packed buffers and allocate memory on GPU
    copyToGPU(gpuValBasePtr, valBufferPacked, valIdx);
}

//#define NDEBUG
template <typename Ftype>
__device__ int_t xlpanelGPU_t<Ftype>::find(int_t k)
{
    #ifdef HAVE_SYCL
    int threadId = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    sycl::group wrk_grp = sycl::ext::oneapi::this_work_item::get_work_group<3>();
    int& idx = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(wrk_grp);
    int& found = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(wrk_grp);
    int nThreads = wrk_grp.get_local_range(2);
    #else
    int threadId = threadIdx_x;
    __shared__ int idx;
    __shared__ int found;
    int nThreads = blockDim_x;
    #endif
    if (!threadId)
    {
        idx = -1;
        found = 0;
    }

    int blocksPerThreads = CEILING(nblocks(), nThreads);
    __syncthreads();
    for (int blk = blocksPerThreads * threadId;
         blk < blocksPerThreads * (threadId + 1);
         blk++)
    {
        if(found) break;

        if (blk < nblocks())
        {
            if (k == gid(blk))
            {
                idx = blk;
                found = 1;
            }
        }
    }
    __syncthreads();
    return idx;
}

template <typename Ftype>
__device__ int_t xupanelGPU_t<Ftype>::find(int_t k)
{
    #ifdef HAVE_SYCL
    int threadId = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    sycl::group wrk_grp = sycl::ext::oneapi::this_work_item::get_work_group<3>();
    int& idx = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(wrk_grp);
    int& found = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(wrk_grp);
    int nThreads = wrk_grp.get_local_range(2);
    #else
    int threadId = threadIdx_x;
    __shared__ int idx;
    __shared__ int found;
    int nThreads = blockDim_x;
    #endif
    if (!threadId)
    {
        idx = -1;
        found = 0;
    }
    __syncthreads();

    int blocksPerThreads = CEILING(nblocks(), nThreads);

    for (int blk = blocksPerThreads * threadId;
         blk < blocksPerThreads * (threadId + 1);
         blk++)
    {
        if(found) break;

        if (blk < nblocks())
        {
            if (k == gid(blk))
            {
                idx = blk;
                found = 1;
            }
        }
    }
    __syncthreads();
    return idx;
}

__device__ int computeIndirectMapGPU(int *rcS2D, int_t srcLen, int_t *srcVec,
                                     int_t dstLen, int_t *dstVec,
                                     int *dstIdx)
{
    #ifdef HAVE_SYCL
    int threadId = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2);
    #else
    int threadId = threadIdx_x;
    #endif
    if (dstVec == NULL) /*uncompressed dimension*/
    {
        if (threadId < srcLen)
            rcS2D[threadId] = srcVec[threadId];
        __syncthreads();
        return 0;
    }

    if (threadId < dstLen)
        dstIdx[dstVec[threadId]] = threadId;
    __syncthreads();

    if (threadId < srcLen)
        rcS2D[threadId] = dstIdx[srcVec[threadId]];
    __syncthreads();

    return 0;
}

template <typename Ftype>
__device__ void scatterGPU_dev(
    int iSt, int jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype>& lpanel, xupanelGPU_t<Ftype>& upanel,
    xLUstructGPU_t<Ftype> *dA
#ifdef HAVE_SYCL
    , sycl::nd_item<3>& item,
    int* baseSharedPtr
#endif
)
{
    // calculate gi,gj
    int ii = iSt + blockIdx_x;
    int jj = jSt + blockIdx_y;
    int threadId = threadIdx_x;

    int gi = lpanel.gid(ii);
    int gj = upanel.gid(jj);
#ifndef NDEBUG
    // if (!threadId)
    //     printf("Scattering to (%d, %d) \n", gi, gj);
#endif
    Ftype *Dst;
    int_t lddst;
    int_t dstRowLen, dstColLen;
    int_t *dstRowList;
    int_t *dstColList;
    int li, lj;
    if (gj > gi) // its in upanel
    {
        li = dA->g2lRow(gi);
        lj = dA->uPanelVec[li].find(gj);
        Dst = dA->uPanelVec[li].blkPtr(lj);
        lddst = dA->supersize(gi);
        dstRowLen = dA->supersize(gi);
        dstRowList = NULL;
        dstColLen = dA->uPanelVec[li].nbcol(lj);
        dstColList = dA->uPanelVec[li].colList(lj);
    }
    else
    {
        lj = dA->g2lCol(gj);
        li = dA->lPanelVec[lj].find(gi);
        Dst = dA->lPanelVec[lj].blkPtr(li);
        lddst = dA->lPanelVec[lj].LDA();
        dstRowLen = dA->lPanelVec[lj].nbrow(li);
        dstRowList = dA->lPanelVec[lj].rowList(li);
        // if(!threadId )
        // printf("Scattering to (%d, %d) by %d li=%d\n",gi, gj,threadId,li);
        dstColLen = dA->supersize(gj);
        dstColList = NULL;
    }

    // compute source row to dest row mapping
    int maxSuperSize = dA->maxSuperSize;
    #ifndef HAVE_SYCL
    extern __shared__ int baseSharedPtr[];
    #endif
    int *rowS2D = baseSharedPtr;
    int *colS2D = &rowS2D[maxSuperSize];
    int *dstIdx = &colS2D[maxSuperSize];

    int nrows = lpanel.nbrow(ii);
    int ncols = upanel.nbcol(jj);
    // lpanel.rowList(ii), upanel.colList(jj)

    computeIndirectMapGPU(rowS2D, nrows, lpanel.rowList(ii),
                          dstRowLen, dstRowList, dstIdx);

    // compute source col to dest col mapping
    computeIndirectMapGPU(colS2D, ncols, upanel.colList(jj),
                          dstColLen, dstColList, dstIdx);

    int nThreads = blockDim_x;
    int colsPerThreadBlock = nThreads / nrows;

    int rowOff = lpanel.stRow(ii) - lpanel.stRow(iSt);
    int colOff = upanel.stCol(jj) - upanel.stCol(jSt);
    Ftype *Src = &gemmBuff[rowOff + colOff * LDgemmBuff];
    int ldsrc = LDgemmBuff;
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
            atomicAddT<Ftype>(&Dst[rowS2D[i] + lddst * colS2D[j]], -Src[i + ldsrc * j]);
#else
            Dst[rowS2D[i] + lddst * colS2D[j]] -= Src[i + ldsrc * j];
#endif
            j += colsPerThreadBlock;
        }
    }

    __syncthreads();
}
template <typename Ftype>
__global__ void scatterGPU(
    int iSt, int jSt,
    Ftype *gemmBuff, int LDgemmBuff,
    xlpanelGPU_t<Ftype> lpanel, xupanelGPU_t<Ftype> upanel,
    xLUstructGPU_t<Ftype> *dA
#ifdef HAVE_SYCL
    , sycl::nd_item<3>& item, int* baseSharedPtr // looks like a bug since the size is `int_t`
#endif
                           )
{
    scatterGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA
                   #ifdef HAVE_SYCL
                   , item, baseSharedPtr
                   #endif
                   );
}

template <typename Ftype>
__global__ void scatterGPU_batch(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch,
    Ftype **gemmBuff_ptrs, int *LDgemmBuff_batch, xlpanelGPU_t<Ftype> *lpanels,
    xupanelGPU_t<Ftype> *upanels, xLUstructGPU_t<Ftype> *dA
#ifdef HAVE_SYCL
    , sycl::nd_item<3>& item, int* baseSharedPtr
#endif
)
{
    int batch_index = blockIdx_z;
    int iSt = iSt_batch[batch_index], iEnd = iEnd_batch[batch_index];
    int jSt = jSt_batch[batch_index], jEnd = jEnd_batch[batch_index];

    int ii = iSt + blockIdx_x;
    int jj = jSt + blockIdx_y;
    if(ii >= iEnd || jj >= jEnd)
        return;

    Ftype* gemmBuff = gemmBuff_ptrs[batch_index];
    if(gemmBuff == NULL)
        return;

    int LDgemmBuff = LDgemmBuff_batch[batch_index];
    lpanelGPU_t& lpanel = lpanels[batch_index];
    upanelGPU_t& upanel = upanels[batch_index];
    scatterGPU_dev(iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA
                   #ifdef HAVE_SYCL
                   , item, baseSharedPtr
                   #endif
                   );
}

template <typename Ftype>
void scatterGPU_driver(
    int iSt, int iEnd, int jSt, int jEnd, Ftype *gemmBuff, int LDgemmBuff,
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> lpanel, xupanelGPU_t<Ftype> upanel,
    xLUstructGPU_t<Ftype> *dA, gpuStream_t cuStream
)
{
    #ifdef HAVE_SYCL
    sycl::range<3> dimBlock(1, 1, ldt); // 1d thread
    sycl::range<3> dimGrid(1, jEnd - jSt, iEnd - iSt);

    cuStream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int_t, 1> localmem(sycl::range<1>(3 * maxSuperSize), cgh);
      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](auto item) {
        scatterGPU<Ftype>(iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA, item,
                          localmem.get_multi_ptr<sycl::access::decorated::yes>().get());
      }); });
    #else
    dim3 dimBlock(ldt); // 1d thread
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

    scatterGPU<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        iSt, jSt, gemmBuff, LDgemmBuff, lpanel, upanel, dA
    );

    checkGPU(gpuGetLastError());
    #endif
}

template <typename Ftype>
void scatterGPU_batchDriver(
    int* iSt_batch, int *iEnd_batch, int *jSt_batch, int *jEnd_batch,
    int max_ilen, int max_jlen, Ftype **gemmBuff_ptrs, int *LDgemmBuff_batch,
    int maxSuperSize, int ldt, xlpanelGPU_t<Ftype> *lpanels, xupanelGPU_t<Ftype> *upanels,
    xLUstructGPU_t<Ftype> *dA, int batchCount, gpuStream_t cuStream
)
{
    const int op_increment = 65535;

    for(int op_start = 0; op_start < batchCount; op_start += op_increment)
	{
        int batch_size = std::min(op_increment, batchCount - op_start);

        #ifdef HAVE_SYCL
        sycl::range<3> dimBlock(1, 1, ldt); // 1d thread
        sycl::range<3> dimGrid(batch_size, max_jlen, max_ilen);

        cuStream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int_t, 1> localmem(sycl::range<1>(3 * maxSuperSize), cgh);
          cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](auto item) {
            scatterGPU_batch<Ftype>(
                iSt_batch + op_start, iEnd_batch + op_start, jSt_batch + op_start,
                jEnd_batch + op_start, gemmBuff_ptrs + op_start, LDgemmBuff_batch + op_start,
                lpanels + op_start, upanels + op_start, dA, item,
                localmem.get_multi_ptr<sycl::access::decorated::yes>().get());
          }); });
        #else
        dim3 dimBlock(ldt); // 1d thread
        dim3 dimGrid(max_ilen, max_jlen, batch_size);
        size_t sharedMemorySize = 3 * maxSuperSize * sizeof(int_t);

        scatterGPU_batch<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
            iSt_batch + op_start, iEnd_batch + op_start, jSt_batch + op_start,
            jEnd_batch + op_start, gemmBuff_ptrs + op_start, LDgemmBuff_batch + op_start,
            lpanels + op_start, upanels + op_start, dA
        );
        #endif
    }
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurComplementUpdateGPU(
    int streamId,
    int_t k, // the k-th panel or supernode
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{

    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int iSt = st_lb;
    int iEnd = iSt;

    int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
    int ncols = upanel.nzcols();

    int maxGemmRows = nrows;
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if (nrows * ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
        maxGemmRows = nrows / numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
        /* printf("buffer exceeded! k = %d, st_lb = %d, nlb = %d, nrowsXncols %d, maxGemRows %d, maxGemmCols %d\n",
	   k, st_lb, nlb, nrows*ncols, maxGemmRows, maxGemmCols);*/
    }

    while (iEnd < nlb)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);

        assert(iEnd > iSt);
        int jSt = 0;
        int jEnd = 0;
        while (jEnd < nub)
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            assert(jEnd > jSt);
            gpuStream_t cuStream = A_gpu.cuStreams[streamId];
            #ifndef HAVE_SYCL
            gpublasHandle_t handle = A_gpu.cuHandles[streamId];
            gpublasSetStream(handle, cuStream);
            #endif
            int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
            int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
            int gemm_k = supersize(k);
#if 0
            printf("k = %d, iSt = %d, iEnd = %d, jst = %d, jend = %d\n", k, iSt, iEnd, jSt, jEnd);
            printf("m=%d, n=%d, k=%d\n", gemm_m, gemm_n, gemm_k);
	    fflush(stdout);
#endif

            Ftype alpha = one<Ftype>();
            Ftype beta = zeroT<Ftype>();
            // gpublasDgemm(handle, GPUBLAS_OP_N, GPUBLAS_OP_N,
            //             gemm_m, gemm_n, gemm_k, &alpha,
            //             lpanel.blkPtrGPU(iSt), lpanel.LDA(),
            //             upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
            //             A_gpu.gpuGemmBuffs[streamId], gemm_m);
            #ifdef HAVE_SYCL
            if constexpr (std::is_same_v<Ftype, doublecomplex>) {
              oneapi::mkl::blas::column_major::gemm(*cuStream, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                                    gemm_m, gemm_n, gemm_k, gpuDoubleComplex(alpha.r, alpha.i),
                                                    reinterpret_cast<const gpuDoubleComplex*>(lpanel.blkPtrGPU(iSt)), lpanel.LDA(),
                                                    reinterpret_cast<const gpuDoubleComplex*>(upanel.blkPtrGPU(jSt)), upanel.LDA(),
                                                    gpuDoubleComplex(beta.r, beta.i),
                                                    reinterpret_cast<gpuDoubleComplex*>(A_gpu.gpuGemmBuffs[streamId]), gemm_m);
            } else {
              oneapi::mkl::blas::column_major::gemm(*cuStream, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                                    gemm_m, gemm_n, gemm_k, alpha,
                                                    lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                                                    upanel.blkPtrGPU(jSt), upanel.LDA(), beta,
                                                    A_gpu.gpuGemmBuffs[streamId], gemm_m);
            }
            #else
            myGpublasGemm<Ftype>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                gemm_m, gemm_n, gemm_k, &alpha,
                                lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                                upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                                A_gpu.gpuGemmBuffs[streamId], gemm_m);
            #endif

            scatterGPU_driver<Ftype>(
                iSt, iEnd, jSt, jEnd, A_gpu.gpuGemmBuffs[streamId], gemm_m,
                A_gpu.maxSuperSize, ldt, lpanel.gpuPanel, upanel.gpuPanel,
                dA_gpu, cuStream
            );
        }
    }
    checkGPU(gpuStreamSynchronize(A_gpu.cuStreams[streamId]));
    return 0;
} /* end dSchurComplementUpdateGPU */

template <typename Ftype>
int_t xLUstruct_t<Ftype>::lookAheadUpdateGPU(
    int streamId,
    int_t k, int_t laIdx, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t laILoc = lpanel.find(laIdx);
    int_t laJLoc = upanel.find(laIdx);

    int iSt = st_lb;
    int jSt = 0;

    /* call look ahead update on Lpanel*/
#ifndef HAVE_SYCL // CUDA, HIP
    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        dSchurCompUpdatePartGPU(
            iSt, nlb, laJLoc, laJLoc + 1,
            k, lpanel, upanel,
            A_gpu.lookAheadLStream[streamId],
            A_gpu.lookAheadLGemmBuffer[streamId],
            A_gpu.lookAheadLHandle[streamId]);

    /* call look ahead update on Upanel*/
    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, jSt, laJLoc,
            k, lpanel, upanel,
            A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId],
            A_gpu.lookAheadUHandle[streamId]);
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, laJLoc + 1, nub,
            k, lpanel, upanel,
            A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId],
            A_gpu.lookAheadUHandle[streamId]);
    }
#else
    if (laJLoc != GLOBAL_BLOCK_NOT_FOUND)
        dSchurCompUpdatePartGPU(
            iSt, nlb, laJLoc, laJLoc + 1,
            k, lpanel, upanel,
            A_gpu.lookAheadLStream[streamId],
            A_gpu.lookAheadLGemmBuffer[streamId]);

    /* call look ahead update on Upanel*/
    if (laILoc != GLOBAL_BLOCK_NOT_FOUND)
    {
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, jSt, laJLoc,
            k, lpanel, upanel,
            A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
        dSchurCompUpdatePartGPU(
            laILoc, laILoc + 1, laJLoc + 1, nub,
            k, lpanel, upanel,
            A_gpu.lookAheadUStream[streamId],
            A_gpu.lookAheadUGemmBuffer[streamId]);
    }
#endif // ifndef HAVE_SYCL

    // checkCudaLocal(gpuStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    // checkCudaLocal(gpuStreamSynchronize(A_gpu.lookAheadUStream[streamId]));

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::SyncLookAheadUpdate(int streamId)
{
    checkGPU(gpuStreamSynchronize(A_gpu.lookAheadLStream[streamId]));
    checkGPU(gpuStreamSynchronize(A_gpu.lookAheadUStream[streamId]));

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdateExcludeOneGPU(
    int streamId,
    int_t k, int_t ex, // suypernodes to be excluded
    xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{
    if (lpanel.isEmpty() || upanel.isEmpty())
        return 0;

    int_t st_lb = 0;
    if (myrow == krow(k))
        st_lb = 1;

    int_t nlb = lpanel.nblocks();
    int_t nub = upanel.nblocks();

    int_t exILoc = lpanel.find(ex);
    int_t exJLoc = upanel.find(ex);

    dSchurCompUpLimitedMem(
        streamId,
        st_lb, exILoc, 0, exJLoc,
        k, lpanel, upanel);

    dSchurCompUpLimitedMem(
        streamId,
        st_lb, exILoc, exJLoc + 1, nub,
        k, lpanel, upanel);

    int_t nextStI = exILoc + 1;
    if (exILoc == GLOBAL_BLOCK_NOT_FOUND)
        nextStI = st_lb;
    /*
    for j we don't need to change since, if exJLoc == GLOBAL_BLOCK_NOT_FOUND =-1
    then exJLoc+1 =0 will work out correctly as starting j
    */
    dSchurCompUpLimitedMem(
        streamId,
        nextStI, nlb, 0, exJLoc,
        k, lpanel, upanel);

    dSchurCompUpLimitedMem(
        streamId,
        nextStI, nlb, exJLoc + 1, nub,
        k, lpanel, upanel);

    // checkCudaLocal(gpuStreamSynchronize(A_gpu.cuStreams[streamId]));
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpdatePartGPU(
    int_t iSt, int_t iEnd, int_t jSt, int_t jEnd,
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel,
    gpuStream_t cuStream, Ftype *gemmBuff
    #ifndef HAVE_SYCL
    , gpublasHandle_t handle
    #endif
                                                  )
{
    if (iSt >= iEnd || jSt >= jEnd)
        return 0;

    #ifndef HAVE_SYCL
    gpublasSetStream(handle, cuStream);
    #endif
    int gemm_m = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    int gemm_n = upanel.stCol(jEnd) - upanel.stCol(jSt);
    int gemm_k = supersize(k);
    Ftype alpha = one<Ftype>();
    Ftype beta = zeroT<Ftype>();
#ifndef NDEBUG
   // printf("m=%d, n=%d, k=%d\n", gemm_m, gemm_n, gemm_k);
#endif

    #ifdef HAVE_SYCL
    if constexpr (std::is_same_v<Ftype, doublecomplex>) {
      oneapi::mkl::blas::column_major::gemm(*cuStream, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                            gemm_m, gemm_n, gemm_k, gpuDoubleComplex(alpha.r, alpha.i),
                                            reinterpret_cast<const gpuDoubleComplex*>(lpanel.blkPtrGPU(iSt)), lpanel.LDA(),
                                            reinterpret_cast<const gpuDoubleComplex*>(upanel.blkPtrGPU(jSt)), upanel.LDA(),
                                            gpuDoubleComplex(beta.r, beta.i),
                                            reinterpret_cast<gpuDoubleComplex*>(gemmBuff), gemm_m);
    } else {
      oneapi::mkl::blas::column_major::gemm(*cuStream, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                            gemm_m, gemm_n, gemm_k, alpha,
                                            lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                                            upanel.blkPtrGPU(jSt), upanel.LDA(), beta,
                                            gemmBuff, gemm_m);
    }
    #else
    myGpublasGemm<Ftype>(handle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                        gemm_m, gemm_n, gemm_k, &alpha,
                        lpanel.blkPtrGPU(iSt), lpanel.LDA(),
                        upanel.blkPtrGPU(jSt), upanel.LDA(), &beta,
                        gemmBuff, gemm_m);
    #endif

    // setting up scatter
    #ifdef HAVE_SYCL
    sycl::range<3> dimBlock(1, 1, ldt); // 1d thread
    sycl::range<3> dimGrid(1, jEnd - jSt, iEnd - iSt);
    auto local_dA_gpu = this->dA_gpu;
    cuStream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int_t, 1> localmem(sycl::range<1>(3 * A_gpu.maxSuperSize), cgh);
      cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](auto item) {
        scatterGPU<Ftype>(iSt, jSt, gemmBuff, gemm_m, lpanel.gpuPanel, upanel.gpuPanel, local_dA_gpu,
                          item, localmem.get_multi_ptr<sycl::access::decorated::yes>().get());
      }); });
    #else
    dim3 dimBlock(ldt); // 1d thread
    dim3 dimGrid(iEnd - iSt, jEnd - jSt);
    size_t sharedMemorySize = 3 * A_gpu.maxSuperSize * sizeof(int_t);

    scatterGPU<Ftype><<<dimGrid, dimBlock, sharedMemorySize, cuStream>>>(
        iSt, jSt,
        gemmBuff, gemm_m,
        lpanel.gpuPanel, upanel.gpuPanel, dA_gpu);
    #endif

    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::dSchurCompUpLimitedMem(
    int streamId,
    int_t lStart, int_t lEnd,
    int_t uStart, int_t uEnd,
    int_t k, xlpanel_t<Ftype> &lpanel, xupanel_t<Ftype> &upanel)
{

    if (lStart >= lEnd || uStart >= uEnd)
        return 0;
    int iSt = lStart;
    int iEnd = iSt;
    int nrows = lpanel.stRow(lEnd) - lpanel.stRow(lStart);
    int ncols = upanel.stCol(uEnd) - upanel.stCol(uStart);

    int maxGemmRows = nrows;
    int maxGemmCols = ncols;
    // entire gemm doesn't fit in gemm buffer
    if (nrows * ncols > A_gpu.gemmBufferSize)
    {
        int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
        int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
        maxGemmRows = nrows / numberofRowChunks;
        maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
    }

    while (iEnd < lEnd)
    {
        iSt = iEnd;
        iEnd = lpanel.getEndBlock(iSt, maxGemmRows);
        if (iEnd > lEnd)
            iEnd = lEnd;

        assert(iEnd > iSt);
        int jSt = uStart;
        int jEnd = uStart;
        while (jEnd < uEnd)
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols);
            if (jEnd > uEnd)
                jEnd = uEnd;

            gpuStream_t cuStream = A_gpu.cuStreams[streamId];
            #ifndef HAVE_SYCL
            gpublasHandle_t handle = A_gpu.cuHandles[streamId];
            dSchurCompUpdatePartGPU(iSt, iEnd, jSt, jEnd,
                                    k, lpanel, upanel, cuStream, A_gpu.gpuGemmBuffs[streamId], handle);
            #else
            dSchurCompUpdatePartGPU(iSt, iEnd, jSt, jEnd,
                                    k, lpanel, upanel, cuStream, A_gpu.gpuGemmBuffs[streamId]);
            #endif
        }
    }

    return 0;
}

int getMPIProcsPerGPU()
{
    if (!(getenv("MPI_PROCESS_PER_GPU")))
    {
        return 1;
    } else {
        int devCount = 0;
        gpuGetDeviceCount(&devCount);
        int envCount = atoi(getenv("MPI_PROCESS_PER_GPU"));
        envCount = SUPERLU_MAX(envCount, 1);
        printf("MPI_PROCESS_PER_GPU=%d, devCount=%d\n", envCount, devCount);
        return SUPERLU_MIN(envCount, devCount);
    }
}

size_t getGPUMemPerProcs(MPI_Comm baseCommunicator)
{
    size_t mfree, mtotal;
    // TODO: shared memory communicator should be part of
    //  LU struct
    //  MPI_Comm sharedComm;
    //  MPI_Comm_split_type(baseCommunicator, MPI_COMM_TYPE_SHARED,
    //                      0, MPI_INFO_NULL, &sharedComm);
    //  MPI_Barrier(sharedComm);
    gpuMemGetInfo(&mfree, &mtotal);
    // MPI_Barrier(sharedComm);
    // MPI_Comm_free(&sharedComm);
#if 0
    printf("Total memory %zu & free memory %zu\n", mtotal, mfree);
#endif

    return (size_t)(USABLE_GPU_MEM_FRACTION * (double)mfree) / getMPIProcsPerGPU();
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::setLUstruct_GPU()
{
    int i, stream;

#if (DEBUGlevel >= 1)
    int iam = 0;
    CHECK_MALLOC(iam, "Enter setLUstruct_GPU()"); fflush(stdout);
#endif

    A_gpu.Pr = Pr;
    A_gpu.Pc = Pc;
    A_gpu.maxSuperSize = ldt;

    /* Sherry: this mapping may be inefficient on Frontier */
    /*Mapping to device*/
    int deviceCount = 0;
    gpuGetDeviceCount(&deviceCount); // How many GPUs?
    printf("gpu count on the node: %d\n", deviceCount);
    int device_id = grid3d->iam % deviceCount;
    gpuSetDevice(device_id);

    double tRegion[5];
    size_t useableGPUMem = getGPUMemPerProcs(grid3d->comm);
    /**
     *  Memory is divided into two parts data memory and buffer memory
     *  data memory is used for useful data
     *  bufferMemory is used for buffers
     * */
    size_t memReqData = 0;

    /*Memory for XSUP*/
    memReqData += (nsupers + 1) * sizeof(int_t);

    tRegion[0] = SuperLU_timer_();

    size_t totalNzvalSize = 0; /* too big for gemmBufferSize */
    size_t max_gemmCsize = 0;  /* Sherry added 2/20/2023 */
    size_t max_nzrow = 0;  /* Yang added 10/20/2023 */
    size_t max_nzcol = 0;

    /*Memory for lpapenl and upanel Data*/
    for (i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            memReqData += lPanelVec[i].totalSize();
            totalNzvalSize += lPanelVec[i].nzvalSize();
            if(lPanelVec[i].nzvalSize()>0)
                max_nzrow = SUPERLU_MAX(lPanelVec[i].nzrows(),max_nzrow);
	    //max_gemmCsize = SUPERoLU_MAX(max_gemmCsize, ???);
        }
    }
    for (i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            memReqData += uPanelVec[i].totalSize();
            totalNzvalSize += uPanelVec[i].nzvalSize();
            if(uPanelVec[i].nzvalSize()>0)
                max_nzcol = SUPERLU_MAX(uPanelVec[i].nzcols(),max_nzcol);
        }
    }
    max_gemmCsize = max_nzcol*max_nzrow;

    memReqData += CEILING(nsupers, Pc) * sizeof(lpanelGPU_t);
    memReqData += CEILING(nsupers, Pr) * sizeof(upanelGPU_t);

    memReqData += sizeof(xLUstructGPU_t<Ftype>);

    // Per stream data
    // TODO: estimate based on ancestor size
    int_t maxBuffSize = sp_ienv_dist (8, options);
    int maxsup = sp_ienv_dist(3, options); // max. supernode size
    maxBuffSize = SUPERLU_MAX(maxsup * maxsup, maxBuffSize); // Sherry added 7/10/23

 #if 0
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, totalNzvalSize);
 #else
    A_gpu.gemmBufferSize = SUPERLU_MIN(maxBuffSize, SUPERLU_MAX(max_gemmCsize,totalNzvalSize)); /* Yang added 10/20/2023 */
 #endif

    size_t dataPerStream = 3 * sizeof(Ftype) * maxLvalCount + 3 * sizeof(Ftype) * maxUvalCount + 2 * sizeof(int_t) * maxLidxCount + 2 * sizeof(int_t) * maxUidxCount + A_gpu.gemmBufferSize * sizeof(Ftype) + ldt * ldt * sizeof(Ftype);
    if (memReqData + 2 * dataPerStream > useableGPUMem)
    {
        printf("Not enough memory on GPU: available = %zu, required for 2 streams =%zu, exiting\n", useableGPUMem, memReqData + 2 * dataPerStream);
        exit(-1);
    }

    tRegion[0] = SuperLU_timer_() - tRegion[0];
#if ( PRNTlevel>=1 )
    // print the time taken to estimate memory on GPU
    if (grid3d->iam == 0)
    {
        printf("GPU deviceCount=%d\n", deviceCount);
	printf("\t.. totalNzvalSize %ld, gemmBufferSize %ld\n",
	       (long) totalNzvalSize, (long) A_gpu.gemmBufferSize);
    }
#endif

    /*Memory for lapenlPanel Data*/
    tRegion[1] = SuperLU_timer_();

    int_t maxNumberOfStream = (useableGPUMem - memReqData) / dataPerStream;

    int numberOfStreams = SUPERLU_MIN(getNumLookAhead(options), maxNumberOfStream);
    numberOfStreams = SUPERLU_MIN(numberOfStreams, MAX_GPU_STREAMS);
    int rNumberOfStreams;
    MPI_Allreduce(&numberOfStreams, &rNumberOfStreams, 1,
                  MPI_INT, MPI_MIN, grid3d->comm);
    A_gpu.numGpuStreams = rNumberOfStreams;

#if ( PRNTlevel>=1 )
    if (!grid3d->iam)
        printf("Using %d GPU LookAhead streams\n", rNumberOfStreams);
    // size_t totalMemoryRequired = memReqData + numberOfStreams * dataPerStream;
#endif

#if 0 /**** Old code ****/
    upanelGPU_t *uPanelVec_GPU = new upanelGPU_t[CEILING(nsupers, Pr)];
    lpanelGPU_t *lPanelVec_GPU = new lpanelGPU_t[CEILING(nsupers, Pc)];
    void *gpuBasePtr, *gpuCurrentPtr;
    gpuMalloc((void**)&gpuBasePtr, totalMemoryRequired);
    gpuCurrentPtr = gpuBasePtr;

    A_gpu.xsup = (int_t *)gpuCurrentPtr;
    gpuCurrentPtr = (int_t *)gpuCurrentPtr + (nsupers + 1);
    gpuMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), gpuMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + lPanelVec[i].totalSize();
        }
    }
    A_gpu.lPanelVec = (xlpanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>);
    gpuMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
               CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>), gpuMemcpyHostToDevice);

    for (int i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU(gpuCurrentPtr);
            gpuCurrentPtr = (char *)gpuCurrentPtr + uPanelVec[i].totalSize();
        }
    }
    A_gpu.uPanelVec = (xupanelGPU_t<Ftype> *)gpuCurrentPtr;
    gpuCurrentPtr = (char *)gpuCurrentPtr + CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>);
    gpuMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
               CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>), gpuMemcpyHostToDevice);

    for (int stream = 0; stream < A_gpu.numGpuStreams; stream++)
    {

        gpuStreamCreate(&A_gpu.cuStreams[stream]);
        gpuStreamCreate(&A_gpu.lookAheadLStream[stream]);
        gpuStreamCreate(&A_gpu.lookAheadUStream[stream]);
        #ifndef HAVE_SYCL
        gpublasCreate(&A_gpu.cuHandles[stream]);
        gpublasCreate(&A_gpu.lookAheadLHandle[stream]);
        gpublasCreate(&A_gpu.lookAheadUHandle[stream]);
        #endif

        A_gpu.LvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        A_gpu.UvalRecvBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
        A_gpu.LidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxLidxCount;
        A_gpu.UidxRecvBufs[stream] = (int_t *)gpuCurrentPtr;
        gpuCurrentPtr = (int_t *)gpuCurrentPtr + maxUidxCount;

        A_gpu.gpuGemmBuffs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + A_gpu.gemmBufferSize;
        A_gpu.dFBufs[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + ldt * ldt;

        /*lookAhead buffers and stream*/
        A_gpu.lookAheadLGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxLvalCount;
        A_gpu.lookAheadUGemmBuffer[stream] = (Ftype *)gpuCurrentPtr;
        gpuCurrentPtr = (Ftype *)gpuCurrentPtr + maxUvalCount;
    }
    // cudaCheckError();
    // allocate
    dA_gpu = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr;

    gpuMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), gpuMemcpyHostToDevice);
    gpuCurrentPtr = (xLUstructGPU_t<Ftype> *)gpuCurrentPtr + 1;

#else /* else of #if 0 ----> this is the current active code - Sherry */
    checkGPU(gpuMalloc((void**)&A_gpu.xsup, (nsupers + 1) * sizeof(int_t)));
    checkGPU(gpuMemcpy(A_gpu.xsup, xsup, (nsupers + 1) * sizeof(int_t), gpuMemcpyHostToDevice));

    double tLsend, tUsend;
#if 0
    tLsend = SuperLU_timer_();
    xupanelGPU_t<Ftype> *uPanelVec_GPU = copyUpanelsToGPU();
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = copyLpanelsToGPU();
    tUsend = SuperLU_timer_() - tUsend;
#else
    xupanelGPU_t<Ftype> *uPanelVec_GPU = new xupanelGPU_t<Ftype>[CEILING(nsupers, Pr)];
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = new xlpanelGPU_t<Ftype>[CEILING(nsupers, Pc)];
    tLsend = SuperLU_timer_();
    for (i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec_GPU[i] = lPanelVec[i].copyToGPU();
    }
    tLsend = SuperLU_timer_() - tLsend;
    tUsend = SuperLU_timer_();
    // cudaCheckError();
    for (i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec_GPU[i] = uPanelVec[i].copyToGPU();
    }
    tUsend = SuperLU_timer_() - tUsend;
#endif
    tRegion[1] = SuperLU_timer_() - tRegion[1];

    checkGPU(gpuMalloc((void**)&A_gpu.lPanelVec, CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>)));
    checkGPU(gpuMemcpy(A_gpu.lPanelVec, lPanelVec_GPU,
                       CEILING(nsupers, Pc) * sizeof(xlpanelGPU_t<Ftype>), gpuMemcpyHostToDevice));
    checkGPU(gpuMalloc((void**)&A_gpu.uPanelVec, CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>)));
    checkGPU(gpuMemcpy(A_gpu.uPanelVec, uPanelVec_GPU,
                       CEILING(nsupers, Pr) * sizeof(xupanelGPU_t<Ftype>), gpuMemcpyHostToDevice));

    delete [] uPanelVec_GPU;
    delete [] lPanelVec_GPU;

    tRegion[2] = SuperLU_timer_();
    int dfactBufSize = 0;
    // TODO: does it work with NULL pointer?
    #ifdef HAVE_SYCL
    dfactBufSize = oneapi::mkl::lapack::getrf_scratchpad_size<double>(*(sycl_get_queue()), ldt, ldt, ldt);
    #else
    gpusolverDnHandle_t gpusolverH = NULL;
    gpusolverDnCreate(&gpusolverH);
    gpusolverDnDgetrf_bufferSize(gpusolverH, ldt, ldt, NULL, ldt, &dfactBufSize);
    gpusolverDnDestroy(gpusolverH);
    #endif
#if ( PRNTlevel >= 1 )
    printf("Size of dfactBuf is %d\n", dfactBufSize);
#endif
    tRegion[2] = SuperLU_timer_() - tRegion[2];

    tRegion[3] = SuperLU_timer_();

    double tcuMalloc=SuperLU_timer_();

    /* Sherry: where are these freed ?? */
    for (stream = 0; stream < A_gpu.numGpuStreams; stream++)
    {
        checkGPU(gpuMalloc((void**)&A_gpu.LvalRecvBufs[stream], sizeof(Ftype) * maxLvalCount));
        checkGPU(gpuMalloc((void**)&A_gpu.UvalRecvBufs[stream], sizeof(Ftype) * maxUvalCount));
        checkGPU(gpuMalloc((void**)&A_gpu.LidxRecvBufs[stream], sizeof(int_t) * maxLidxCount));
        checkGPU(gpuMalloc((void**)&A_gpu.UidxRecvBufs[stream], sizeof(int_t) * maxUidxCount));
        // allocate the space for diagonal factor on GPU
        checkGPU(gpuMalloc((void**)&A_gpu.diagFactWork[stream], sizeof(Ftype) * dfactBufSize));
        checkGPU(gpuMalloc((void**)&A_gpu.diagFactInfo[stream], sizeof(int)));

        /*lookAhead buffers and stream*/
        checkGPU(gpuMalloc((void**)&A_gpu.lookAheadLGemmBuffer[stream], sizeof(Ftype) * maxLvalCount));

        checkGPU(gpuMalloc((void**)&A_gpu.lookAheadUGemmBuffer[stream], sizeof(Ftype) * maxUvalCount));
	// Sherry: replace this by new code
        //gpuMalloc((void**)&A_gpu.dFBufs[stream], ldt * ldt * sizeof(Ftype));
        //gpuMalloc((void**)&A_gpu.gpuGemmBuffs[stream], A_gpu.gemmBufferSize * sizeof(Ftype));
    }

    /* Sherry: dfBufs[] changed to Ftype pointer **, max(batch, numGpuStreams) */
    int mxLeafNode = trf3Dpartition->mxLeafNode, mx_fsize = 0;

    /* Compute gemmCsize[] for batch operations
       !!!!!!! WARNING: this only works for 1 MPI  !!!!!! */
    if ( options->batchCount > 0 ) {
	trf3Dpartition->gemmCsizes = int32Calloc_dist(mxLeafNode);
	int k, k0, k_st, k_end, offset, Csize;

	for (int ilvl = 0; ilvl < maxLvl; ++ilvl) {  /* Loop through the Pz tree levels */
	    int treeId = trf3Dpartition->myTreeIdxs[ilvl];
	    sForest_t* sforest = trf3Dpartition->sForests[treeId];
	    if (sforest){
		int_t *perm_c_supno = sforest->nodeList ;
                mx_fsize = std::max((int_t)mx_fsize, sforest->nNodes);

		int maxTopoLevel = sforest->topoInfo.numLvl;/* number of levels at each outer-tree node */
		for (int topoLvl = 0; topoLvl < maxTopoLevel; ++topoLvl) {
		    k_st = sforest->topoInfo.eTreeTopLims[topoLvl];
		    k_end = sforest->topoInfo.eTreeTopLims[topoLvl + 1];

		    for (k0 = k_st; k0 < k_end; ++k0) {
			offset = k0 - k_st;
			k = perm_c_supno[k0];
			Csize = lPanelVec[k].nzrows() * uPanelVec[k].nzcols();
			trf3Dpartition->gemmCsizes[offset] =
			    SUPERLU_MAX(trf3Dpartition->gemmCsizes[offset], Csize);
		    }
		}
	    }
	}
    }

    int num_dfbufs;  /* number of diagonal buffers */
    if ( options->batchCount > 0 ) { /* use batch code */
	num_dfbufs = mxLeafNode;
    } else { /* use pipelined code */
	// num_dfbufs = MAX_GPU_STREAMS; //
    num_dfbufs = A_gpu.numGpuStreams;
    }
    int num_gemmbufs = num_dfbufs;
#if ( PRNTlevel >= 1 )
    printf(".. setLUstrut_GPU: num_dfbufs %d, num_gemmbufs %d\n", num_dfbufs, num_gemmbufs);
    fflush(stdout);
#endif

    A_gpu.dFBufs = (Ftype **) SUPERLU_MALLOC(num_dfbufs * sizeof(Ftype *));
    A_gpu.gpuGemmBuffs = (Ftype **) SUPERLU_MALLOC(num_gemmbufs * sizeof(Ftype *));

    int l, sum_diag_size = 0, sum_gemmC_size = 0;

    if ( options->batchCount > 0 ) { /* set up variable-size buffers for batch code */
	for (i = 0; i < num_dfbufs; ++i) {
	    l = trf3Dpartition->diagDims[i];
	    checkGPU(gpuMalloc((void**)&(A_gpu.dFBufs[i]), l * l * sizeof(Ftype)));
	    //printf("\t diagDims[%d] %d\n", i, l);
	    checkGPU(gpuMalloc((void**)&(A_gpu.gpuGemmBuffs[i]), trf3Dpartition->gemmCsizes[i] * sizeof(Ftype)));
	    sum_diag_size += l * l;
	    sum_gemmC_size += trf3Dpartition->gemmCsizes[i];
	}
    } else { /* uniform-size buffers */
	l = ldt * ldt;
	for (i = 0; i < num_dfbufs; ++i) {
        checkGPU(gpuMalloc((void**)&(A_gpu.dFBufs[i]), l * sizeof(Ftype)));
	    checkGPU(gpuMalloc((void**)&(A_gpu.gpuGemmBuffs[i]), A_gpu.gemmBufferSize * sizeof(Ftype)));
	}
    }

    // Wajih: Adding allocation for batched LU and SCU marshalled data
    // TODO: these are serialized workspaces, so the allocations can be shared

#if 0
    A_gpu.marshall_data.setBatchSize(num_dfbufs);
    A_gpu.sc_marshall_data.setBatchSize(num_dfbufs);
#endif

    // TODO: where should these be freed?
    // Allocate GPU copy for the node list
    checkGPU(gpuMalloc((void**)&(A_gpu.dperm_c_supno), sizeof(int) * mx_fsize));
    // Allocate GPU copy of all the gemm buffer pointers and copy the host array to the GPU
    checkGPU(gpuMalloc((void**)&(A_gpu.dgpuGemmBuffs), sizeof(Ftype*) * num_gemmbufs));
    checkGPU(gpuMemcpy(A_gpu.dgpuGemmBuffs, A_gpu.gpuGemmBuffs, sizeof(Ftype*) * num_gemmbufs, gpuMemcpyHostToDevice));

    tcuMalloc = SuperLU_timer_() - tcuMalloc;
#if ( PRNTlevel>=1 )
    printf("Time to allocate GPU memory: %g\n", tcuMalloc);
    printf("\t.. sum_diag_size %d\t sum_gemmC_size %d\n", sum_diag_size, sum_gemmC_size);
    fflush(stdout);
#endif

    double tcuStream=SuperLU_timer_();

    #ifndef HAVE_SYCL
    for (stream = 0; stream < A_gpu.numGpuStreams; stream++)
    {
        // gpublasCreate(&A_gpu.cuHandles[stream]);
        gpusolverDnCreate(&A_gpu.cuSolveHandles[stream]);
    }
    #endif
    tcuStream = SuperLU_timer_() - tcuStream;

    double tcuStreamCreate=SuperLU_timer_();
    for (stream = 0; stream < A_gpu.numGpuStreams; stream++)
    {
        gpuStreamCreate(&A_gpu.cuStreams[stream]);
        /*lookAhead buffers and stream*/
        gpuStreamCreate(&A_gpu.lookAheadLStream[stream]);
        gpuStreamCreate(&A_gpu.lookAheadUStream[stream]);
        #ifndef HAVE_SYCL
        gpublasCreate(&A_gpu.cuHandles[stream]);
        gpublasCreate(&A_gpu.lookAheadLHandle[stream]);
        gpublasCreate(&A_gpu.lookAheadUHandle[stream]);
        #endif

        // Wajih: Need to create at least one magma queue
#ifdef HAVE_MAGMA
        if(stream == 0)
        {
            magma_queue_create_from_cuda(
                device_id, A_gpu.cuStreams[stream], A_gpu.cuHandles[stream],
                NULL, &A_gpu.magma_queue
            );
        }
#endif
    }
    tcuStreamCreate = SuperLU_timer_() - tcuStreamCreate;
    tRegion[3] = SuperLU_timer_() - tRegion[3];

#if ( PRNTlevel >= 1 )
    printf("Time to create gpublas streams: %g\n", tcuStream);
    printf("Time to create GPU streams: %g\n", tcuStreamCreate);
    printf("Time taken to estimate memory on GPU: %f\n", tRegion[0]);
    printf("TRegion L,U send: \t %g\n", tRegion[1]);
    printf("Time to send Lpanel=%g  and U panels =%g \n", tLsend, tUsend);
    printf("TRegion dfactBuf: \t %g\n", tRegion[2]);
    printf("TRegion stream: \t %g\n", tRegion[3]);
    fflush(stdout);
#endif

    // allocate
    checkGPU(gpuMalloc((void**)&dA_gpu, sizeof(xLUstructGPU_t<Ftype>)));
    checkGPU(gpuMemcpy(dA_gpu, &A_gpu, sizeof(xLUstructGPU_t<Ftype>), gpuMemcpyHostToDevice));

#endif /* match #if 0 ... #else ... */

    // cudaCheckError();

#if (DEBUGlevel >= 1)
	CHECK_MALLOC(iam, "Exit setLUstruct_GPU()");
#endif
    return 0;
} /* setLUstruct_GPU */

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUGPUtoHost()
{

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec[i].copyFromGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec[i].copyFromGPU();
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::copyLUHosttoGPU()
{
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
            lPanelVec[i].copyBackToGPU();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            uPanelVec[i].copyBackToGPU();
    return 0;
}

template <typename Ftype>
int_t xLUstruct_t<Ftype>::checkGPUPanel()
{

    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
        lPanelVec[i].checkGPUPanel();

    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        uPanelVec[i].checkGPUPanel();

    std::cout << "Checking LU struct completed succesfully"
              << "\n";
    return 0;
}


/**
 * @brief Pack non-zero values into a vector.
 *
 * @param spNzvalArray The array of non-zero values.
 * @param nzvalSize The size of the array of non-zero values.
 * @param valOffset The offset of the non-zero values.
 * @param packedNzvals The vector to store the non-zero values.
 * @param packedNzvalsIndices The vector to store the indices of the non-zero values.
 */
template <typename Ftype>
void packNzvals(std::vector<Ftype> &packedNzvals, std::vector<int_t> &packedNzvalsIndices,
                Ftype *spNzvalArray, int_t nzvalSize, int_t valOffset)
{
    for (int k = 0; k < nzvalSize; k++)
    {
        if (spNzvalArray[k] != 0)
        {
            packedNzvals.push_back(spNzvalArray[k]);
            packedNzvalsIndices.push_back(valOffset + k);
        }
    }
}

const int AVOID_CPU_NZVAL = 1;
template <typename Ftype>
xlpanelGPU_t<Ftype> *xLUstruct_t<Ftype>::copyLpanelsToGPU()
{
    xlpanelGPU_t<Ftype> *lPanelVec_GPU = new xlpanelGPU_t<Ftype>[CEILING(nsupers, Pc)];

    // TODO: set gpuLvalSize, gpuLidxSize
    gpuLvalSize = 0;
    gpuLidxSize = 0;
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            gpuLvalSize += sizeof(Ftype) * lPanelVec[i].nzvalSize();
            gpuLidxSize += sizeof(int_t) * lPanelVec[i].indexSize();
        }
    }

    Ftype *valBuffer = (Ftype *)SUPERLU_MALLOC(gpuLvalSize);
    int_t *idxBuffer = (int_t *)SUPERLU_MALLOC(gpuLidxSize);

    // allocate memory buffer on GPU
    checkGPU(gpuMalloc((void**)&gpuLvalBasePtr, gpuLvalSize));
    checkGPU(gpuMalloc((void**)&gpuLidxBasePtr, gpuLidxSize));

    size_t valOffset = 0;
    size_t idxOffset = 0;
    Ftype tCopyToCPU = SuperLU_timer_();

    std::vector<Ftype> packedNzvals;
    std::vector<int_t> packedNzvalsIndices;

    // do a memcpy to CPU buffer
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (i * Pc + mycol < nsupers && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            if (lPanelVec[i].isEmpty())
            {
                xlpanelGPU_t<Ftype> ithLpanel(NULL, NULL);
                lPanelVec[i].gpuPanel = ithLpanel;
                lPanelVec_GPU[i] = ithLpanel;
            }
            else
            {
                xlpanelGPU_t<Ftype> ithLpanel(&gpuLidxBasePtr[idxOffset], &gpuLvalBasePtr[valOffset]);
                lPanelVec[i].gpuPanel = ithLpanel;
                lPanelVec_GPU[i] = ithLpanel;
                if (AVOID_CPU_NZVAL)
                    packNzvals<Ftype>(packedNzvals, packedNzvalsIndices, lPanelVec[i].val, lPanelVec[i].nzvalSize(), valOffset);
                else
                    memcpy(&valBuffer[valOffset], lPanelVec[i].val, sizeof(Ftype) * lPanelVec[i].nzvalSize());

                memcpy(&idxBuffer[idxOffset], lPanelVec[i].index, sizeof(int_t) * lPanelVec[i].indexSize());

                valOffset += lPanelVec[i].nzvalSize();
                idxOffset += lPanelVec[i].indexSize();
            }
        }
    }
    tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
    std::cout << "Time to copy-L to CPU: " << tCopyToCPU << "\n";
    // do a gpuMemcpy to GPU
    Ftype tLsend = SuperLU_timer_();
    if (AVOID_CPU_NZVAL)
        copyToGPU(gpuLvalBasePtr, packedNzvals, packedNzvalsIndices);
    else
    {
#if 0
            gpuMemcpy(gpuLvalBasePtr, valBuffer, gpuLvalSize, gpuMemcpyHostToDevice);
#else
        copyToGPU_Sparse(gpuLvalBasePtr, valBuffer, gpuLvalSize);
#endif
    }
    // find
    checkGPU(gpuMemcpy(gpuLidxBasePtr, idxBuffer, gpuLidxSize, gpuMemcpyHostToDevice));
    tLsend = SuperLU_timer_() - tLsend;
    printf("gpuMemcpy time L =%g \n", tLsend);

    SUPERLU_FREE(valBuffer);
    SUPERLU_FREE(idxBuffer);
    return lPanelVec_GPU;
} /* copyLpanelsToGPU */

template <typename Ftype>
xupanelGPU_t<Ftype> *xLUstruct_t<Ftype>::copyUpanelsToGPU()
{
#if (DEBUGlevel >= 1)
    int iam = 0;
    CHECK_MALLOC(iam, "Enter copyUpanelsToGPU()");
#endif

    xupanelGPU_t<Ftype> *uPanelVec_GPU = new xupanelGPU_t<Ftype>[CEILING(nsupers, Pr)];

    gpuUvalSize = 0;
    gpuUidxSize = 0;
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            gpuUvalSize += sizeof(Ftype) * uPanelVec[i].nzvalSize();
            gpuUidxSize += sizeof(int_t) * uPanelVec[i].indexSize();
        }
    }

    // TODO: set gpuUvalSize, gpuUidxSize

    // allocate memory buffer on GPU
    checkGPU(gpuMalloc((void**)&gpuUvalBasePtr, gpuUvalSize));
    checkGPU(gpuMalloc((void**)&gpuUidxBasePtr, gpuUidxSize));

    size_t valOffset = 0;
    size_t idxOffset = 0;

    Ftype tCopyToCPU = SuperLU_timer_();
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            if (uPanelVec[i].isEmpty())
            {
                xupanelGPU_t<Ftype> ithupanel(NULL, NULL);
                uPanelVec[i].gpuPanel = ithupanel;
                uPanelVec_GPU[i] = ithupanel;
            }
        }
    }

    int_t *idxBuffer = NULL;
    if ( gpuUidxSize>0 ) /* Sherry fix: gpuUidxSize can be 0 */
	idxBuffer = (int_t *)SUPERLU_MALLOC(gpuUidxSize);

    if (AVOID_CPU_NZVAL)
    {
        printf("AVOID_CPU_NZVAL is set\n");
        std::vector<Ftype> packedNzvals;
        std::vector<int_t> packedNzvalsIndices;
        for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        {
            if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            {
                if (!uPanelVec[i].isEmpty())
                {

                    xupanelGPU_t<Ftype> ithupanel(&gpuUidxBasePtr[idxOffset], &gpuUvalBasePtr[valOffset]);
                    uPanelVec[i].gpuPanel = ithupanel;
                    uPanelVec_GPU[i] = ithupanel;
                    packNzvals<Ftype>(packedNzvals, packedNzvalsIndices, uPanelVec[i].val,
                               uPanelVec[i].nzvalSize(), valOffset);
                    memcpy(&idxBuffer[idxOffset], uPanelVec[i].index, sizeof(int_t) * uPanelVec[i].indexSize());

                    valOffset += uPanelVec[i].nzvalSize();
                    idxOffset += uPanelVec[i].indexSize();
                }
            }
        }
        tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
        printf("copyU to CPU-buff time = %g\n", tCopyToCPU);

        // do a gpuMemcpy to GPU
        Ftype tLsend = SuperLU_timer_();
        copyToGPU(gpuUvalBasePtr, packedNzvals, packedNzvalsIndices);
        checkGPU(gpuMemcpy(gpuUidxBasePtr, idxBuffer, gpuUidxSize, gpuMemcpyHostToDevice));
        tLsend = SuperLU_timer_() - tLsend;
        printf("gpuMemcpy time U =%g \n", tLsend);
        // SUPERLU_FREE(valBuffer);
    }
    else /* AVOID_CPU_NZVAL not set */
    {
        // do a memcpy to CPU buffer
        Ftype *valBuffer = (Ftype *)SUPERLU_MALLOC(gpuUvalSize);

        for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
        {
            if (i * Pr + myrow < nsupers && isNodeInMyGrid[i * Pr + myrow] == 1)
            {
                if (!uPanelVec[i].isEmpty())
                {

                    xupanelGPU_t<Ftype> ithupanel(&gpuUidxBasePtr[idxOffset], &gpuUvalBasePtr[valOffset]);
                    uPanelVec[i].gpuPanel = ithupanel;
                    uPanelVec_GPU[i] = ithupanel;
                    memcpy(&valBuffer[valOffset], uPanelVec[i].val, sizeof(Ftype) * uPanelVec[i].nzvalSize());
                    memcpy(&idxBuffer[idxOffset], uPanelVec[i].index, sizeof(int_t) * uPanelVec[i].indexSize());

                    valOffset += uPanelVec[i].nzvalSize();
                    idxOffset += uPanelVec[i].indexSize();
                }
            }
        }
        tCopyToCPU = SuperLU_timer_() - tCopyToCPU;
        printf("copyU to CPU-buff time = %g\n", tCopyToCPU);

        // do a gpuMemcpy to GPU
        Ftype tLsend = SuperLU_timer_();
        const int USE_GPU_COPY = 1;
        if (USE_GPU_COPY)
        {
            checkGPU(gpuMemcpy(gpuUvalBasePtr, valBuffer, gpuUvalSize, gpuMemcpyHostToDevice));
        }
        else
            copyToGPU_Sparse(gpuUvalBasePtr, valBuffer, gpuUvalSize);

        checkGPU(gpuMemcpy(gpuUidxBasePtr, idxBuffer, gpuUidxSize, gpuMemcpyHostToDevice));
        tLsend = SuperLU_timer_() - tLsend;
        printf("gpuMemcpy time U =%g \n", tLsend);

        SUPERLU_FREE(valBuffer);
    } /* end else AVOID_CPU_NZVAL not set */

    if ( gpuUidxSize>0 ) /* Sherry fix: gpuUidxSize can be 0 */
	SUPERLU_FREE(idxBuffer);

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Exit copyUpanelsToGPU()");
#endif

    return uPanelVec_GPU;

} /* copyUpanelsToGPU */
//#endif


//////// Rest of the code for batch not used anymore
#if (0)
// Marshall Functors for batched execution
template <typename Ftype>
struct MarshallLUFunc {
    int k_st, *ld_batch, *dim_batch;
    Ftype** diag_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallLUFunc(int k_st, Ftype** diag_ptrs, int *ld_batch, int *dim_batch, xLUstructGPU_t<Ftype> *A_gpu)
    {
        this->k_st = k_st;
        this->ld_batch = ld_batch;
        this->dim_batch = dim_batch;
        this->diag_ptrs = diag_ptrs;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        diag_ptrs[i] = lpanel.blkPtr(0);
        ld_batch[i] = lpanel.LDA();
        dim_batch[i] = SuperSize(k);
    }
};

struct MarshallTRSMUFunc {
    int k_st, *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    Ftype** diag_ptrs, **panel_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallTRSMUFunc(
        int k_st, Ftype** diag_ptrs, int *diag_ld_batch, int *diag_dim_batch, Ftype** panel_ptrs,
        int *panel_ld_batch, int *panel_dim_batch, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        int ksupc = SuperSize(k);

        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lRow(k)];
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        if(!upanel.isEmpty())
        {
            panel_ptrs[i] = upanel.blkPtr(0);
            panel_ld_batch[i] = upanel.LDA();
            panel_dim_batch[i] = upanel.nzcols();
            diag_ptrs[i] = lpanel.blkPtr(0);
            diag_ld_batch[i] = lpanel.LDA();
            diag_dim_batch[i] = ksupc;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }
    }
};

struct MarshallTRSMLFunc {
    int k_st, *diag_ld_batch, *diag_dim_batch, *panel_ld_batch, *panel_dim_batch;
    Ftype** diag_ptrs, **panel_ptrs;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallTRSMLFunc(
        int k_st, Ftype** diag_ptrs, int *diag_ld_batch, int *diag_dim_batch, Ftype** panel_ptrs,
        int *panel_ld_batch, int *panel_dim_batch, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->diag_ptrs = diag_ptrs;
        this->diag_ld_batch = diag_ld_batch;
        this->diag_dim_batch = diag_dim_batch;
        this->panel_ptrs = panel_ptrs;
        this->panel_ld_batch = panel_ld_batch;
        this->panel_dim_batch = panel_dim_batch;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        int ksupc = SuperSize(k);

        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];

        if(!lpanel.isEmpty())
        {
            Ftype *lPanelStPtr = lpanel.blkPtr(0);
            int_t len = lpanel.nzrows();
            if(lpanel.haveDiag())
            {
                lPanelStPtr = lpanel.blkPtr(1);
                len -= lpanel.nbrow(0);
            }
            panel_ptrs[i] = lPanelStPtr;
            panel_ld_batch[i] = lpanel.LDA();
            panel_dim_batch[i] = len;
            diag_ptrs[i] = lpanel.blkPtr(0);
            diag_ld_batch[i] = lpanel.LDA();
            diag_dim_batch[i] = ksupc;
        }
        else
        {
            panel_ptrs[i] = diag_ptrs[i] = NULL;
            panel_ld_batch[i] = diag_ld_batch[i] = 1;
            panel_dim_batch[i] = diag_dim_batch[i] = 0;
        }
    }
};

struct MarshallInitSCUFunc {
    int k_st, *ist, *iend, *maxGemmRows, *maxGemmCols;
    lpanelGPU_t* lpanels;
    upanelGPU_t* upanels;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallInitSCUFunc(
        int k_st, int *ist, int *iend, int *maxGemmRows, int *maxGemmCols,
        lpanelGPU_t* lpanels, upanelGPU_t* upanels, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->maxGemmRows = maxGemmRows;
        this->maxGemmCols = maxGemmCols;
        this->lpanels = lpanels;
        this->upanels = upanels;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        size_t gemmBufferSize = A_gpu->gemmBufferSize;

        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];

        lpanels[i] = lpanel;
        upanels[i] = upanel;

        if(!upanel.isEmpty() && !lpanel.isEmpty())
        {
            int_t st_lb = 1;
            int_t nlb = lpanel.nblocks();
            int_t nub = upanel.nblocks();

            ist[i] = iend[i] = st_lb;
            int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
            int ncols = upanel.nzcols();

            int max_rows = nrows;
            int max_cols = ncols;
            // entire gemm doesn't fit in gemm buffer
            if (nrows * ncols > gemmBufferSize)
            {
                int maxGemmOpSize = (int)sqrt((Ftype)gemmBufferSize);
                int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
                max_rows = nrows / numberofRowChunks;
                max_cols = gemmBufferSize / max_rows;
            }

            maxGemmRows[i] = max_rows;
            maxGemmCols[i] = max_cols;
        }
        else
        {
            ist[i] = iend[i] = 0;
            maxGemmRows[i] = maxGemmCols[i] = 0;
        }
    }
};

struct MarshallSCUOuter_Predicate
{
    __host__ __device__ bool operator()(const int &x)
    {
        return x == 1;
    }
};

struct MarshallSCUOuterFunc {
    int k_st, *ist, *iend, *jst, *jend, *maxGemmRows, *done_flags;
    xLUstructGPU_t<Ftype> *A_gpu;

    MarshallSCUOuterFunc(int k_st, int *ist, int *iend, int *jst, int *jend, int *maxGemmRows, int* done_flags, xLUstructGPU_t<Ftype> *A_gpu)
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->jst = jst;
        this->jend = jend;
        this->maxGemmRows = maxGemmRows;
        this->done_flags = done_flags;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];
        int& iEnd = iend[i];

        // Not done if even one operation still has work to do
        if(!lpanel.isEmpty() && !upanel.isEmpty() && iEnd < lpanel.nblocks())
        {
            int& iSt = ist[i];
            iSt = iEnd;
            iEnd = lpanel.getEndBlock(iSt, maxGemmRows[i]);
            assert(iEnd > iSt);
            jst[i] = jend[i] = 0;
            done_flags[i] = 0;
        }
        else
        {
            done_flags[i] = 1;
        }
    }
};

struct MarshallSCUInner_Predicate
{
    __host__ __device__ bool operator()(const int &x)
    {
        return x == 0;
    }
}

#ifdef HAVE_SYCL
// Note: std::unary_functions/thrust::unary_functions are deprecated in
// C++17 standards and above. So custom-functor was chosen as an alternative.
// Need to adopt a similar solution for the CUDA & HIP backends as well. (in the future)
template<typename T>
struct element_diff
{
    T* st, *end;
    element_diff(T* st, T *end) 
    {
        this->st = st;
        this->end = end;
    }
    
    __device__ T operator()(const T &x) const
    {
        return end[x] - st[x];
    }
};
#else
template<typename T>
struct element_diff : public gputhrust::unary_function<T,T>
{
    T* st, *end;
    element_diff(T* st, T *end) 
    {
        this->st = st;
        this->end = end;
    }
    
    __device__ T operator()(const T &x) const
    {
        return end[x] - st[x];
    }
};
#endif // ifdef HAVE_SYCL


struct MarshallSCUInnerFunc {
    int k_st, *ist, *iend, *jst, *jend, *maxGemmCols;
    xLUstructGPU_t<Ftype> *A_gpu;
    Ftype** A_ptrs, **B_ptrs, **C_ptrs;
    int* lda_array, *ldb_array, *ldc_array, *m_array, *n_array, *k_array;

    MarshallSCUInnerFunc(
        int k_st, int *ist, int *iend, int *jst, int *jend, int *maxGemmCols,
        Ftype** A_ptrs, int* lda_array, Ftype** B_ptrs, int* ldb_array, Ftype **C_ptrs, int *ldc_array,
        int *m_array, int *n_array, int *k_array, xLUstructGPU_t<Ftype> *A_gpu
    )
    {
        this->k_st = k_st;
        this->ist = ist;
        this->iend = iend;
        this->jst = jst;
        this->jend = jend;
        this->maxGemmCols = maxGemmCols;
        this->A_ptrs = A_ptrs;
        this->B_ptrs = B_ptrs;
        this->C_ptrs = C_ptrs;
        this->lda_array = lda_array;
        this->ldb_array = ldb_array;
        this->ldc_array = ldc_array;
        this->m_array = m_array;
        this->n_array = n_array;
        this->k_array = k_array;
        this->A_gpu = A_gpu;
    }

    __device__ void operator()(const unsigned int &i) const
    {
        int k = A_gpu->dperm_c_supno[k_st + i];
        int_t* xsup = A_gpu->xsup;
        lpanelGPU_t &lpanel = A_gpu->lPanelVec[A_gpu->g2lCol(k)];
        upanelGPU_t &upanel = A_gpu->uPanelVec[A_gpu->g2lCol(k)];

        int iSt = ist[i], iEnd = iend[i];
        int& jSt = jst[i], &jEnd = jend[i];

        // Not done if even one operation still has work to do
        if(!lpanel.isEmpty() && !upanel.isEmpty() && jEnd < upanel.nblocks())
        {
            jSt = jEnd;
            jEnd = upanel.getEndBlock(jSt, maxGemmCols[i]);
            assert(jEnd > jSt);

            A_ptrs[i] = lpanel.blkPtr(iSt);
            B_ptrs[i] = upanel.blkPtr(jSt);
            C_ptrs[i] = A_gpu->dgpuGemmBuffs[i];

            lda_array[i] = lpanel.LDA();
            ldb_array[i] = upanel.LDA();
            ldc_array[i] = lpanel.stRow(iEnd) - lpanel.stRow(iSt);

            m_array[i] = ldc_array[i];
            n_array[i] = upanel.stCol(jEnd) - upanel.stCol(jSt);
            k_array[i] = SuperSize(k);
        }
        else
        {
            A_ptrs[i] = B_ptrs[i] = C_ptrs[i] = NULL;
            lda_array[i] = ldb_array[i] = ldc_array[i] = 1;
            m_array[i] = n_array[i] = k_array[i] = 0;
        }
    }
}

// Marshalling routines for batched execution
void xLUstruct_t<Ftype>::marshallBatchedLUData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallLUFunc func(k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array, dA_gpu);

    gputhrust::for_each(
        gputhurst_policy, gputhrust::counting_iterator<int>(0),
        gputhrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *ld_batch = mdata.host_diag_ld_array.data();
    // int *dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];

	// 	if (iam == procIJ(k, k))
	// 	{
    //         assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
	// 		diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
	// 		ld_batch[mdata.batchsize] = lpanel.LDA();
	// 		dim_batch[mdata.batchsize] = SuperSize(k);
	// 		mdata.batchsize++;
	// 	}
    // }

    // mdata.setMaxDiag();
    // // Then copy the marshalled data over to the GPU
    // checkGPU(gpuMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_diag_ld_array, ld_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_diag_dim_array, dim_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
}

void xLUstruct_t<Ftype>::marshallBatchedTRSMUData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallTRSMUFunc func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, dA_gpu
    );

    gputhrust::for_each(
        gputhurst_policy, gputhrust::counting_iterator<int>(0),
        gputhrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **panel_ptrs = mdata.host_panel_ptrs.data();
    // int *panel_ld_batch = mdata.host_panel_ld_array.data();
    // int *panel_dim_batch = mdata.host_panel_dim_array.data();
    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *diag_ld_batch = mdata.host_diag_ld_array.data();
    // int *diag_dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;
	// 	int ksupc = SuperSize(k);

	// 	if (myrow == krow(k))
	// 	{
    //         upanel_t& upanel = uPanelVec[g2lRow(k)];
    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
    //         if(!upanel.isEmpty())
    //         {
    //             assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //             panel_ptrs[mdata.batchsize] = upanel.blkPtrGPU(0);
    //             panel_ld_batch[mdata.batchsize] = upanel.LDA();
    //             panel_dim_batch[mdata.batchsize] = upanel.nzcols();

    //             // Hackathon change: using the original diagonal block instead of the bcast buffer
    //             // diag_ptrs[mdata.batchsize] = A_gpu.dFBufs[buffer_offset];
    //             // diag_ld_batch[mdata.batchsize] = ksupc;
    //             // diag_dim_batch[mdata.batchsize] = ksupc;
    //             diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
    //             diag_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             diag_dim_batch[mdata.batchsize] = ksupc;

    //             mdata.batchsize++;
    //         }
	// 	}
    // }

    // mdata.setMaxDiag();
    // mdata.setMaxPanel();

    // // Then copy the marshalled data over to the GPU
    // checkGPU(gpuMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_diag_ld_array, diag_ld_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_diag_dim_array, diag_dim_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_panel_ptrs, panel_ptrs, mdata.batchsize * sizeof(Ftype*), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_panel_ld_array, panel_ld_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
    // checkGPU(gpuMemcpy(mdata.dev_panel_dim_array, panel_dim_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice));
}

void xLUstruct_t<Ftype>::marshallBatchedTRSMLData(int k_st, int k_end, int_t *perm_c_supno)
{
    // First gather up all the pointer and meta data on the host
    LUMarshallData& mdata = A_gpu.marshall_data;
    mdata.batchsize = k_end - k_st;

    MarshallTRSMLFunc func(
        k_st, mdata.dev_diag_ptrs, mdata.dev_diag_ld_array, mdata.dev_diag_dim_array,
        mdata.dev_panel_ptrs, mdata.dev_panel_ld_array, mdata.dev_panel_dim_array, dA_gpu
    );

    gputhrust::for_each(
        gputhurst_policy, gputhrust::counting_iterator<int>(0),
        gputhrust::counting_iterator<int>(mdata.batchsize), func
    );

    // Ftype **panel_ptrs = mdata.host_panel_ptrs.data();
    // int *panel_ld_batch = mdata.host_panel_ld_array.data();
    // int *panel_dim_batch = mdata.host_panel_dim_array.data();
    // Ftype **diag_ptrs = mdata.host_diag_ptrs.data();
    // int *diag_ld_batch = mdata.host_diag_ld_array.data();
    // int *diag_dim_batch = mdata.host_diag_dim_array.data();

	// mdata.batchsize = 0;

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;
	// 	int ksupc = SuperSize(k);

	// 	if (mycol == kcol(k))
	// 	{
    //         xlpanel_t<Ftype> &lpanel = lPanelVec[g2lCol(k)];
    //         if(!lpanel.isEmpty())
    //         {
    //             assert(mdata.batchsize < mdata.host_diag_ptrs.size());

    //             Ftype *lPanelStPtr = lpanel.blkPtrGPU(0);
    //             int_t len = lpanel.nzrows();
    //             if(lpanel.haveDiag())
    //             {
    //                 /* code */
    //                 lPanelStPtr = lpanel.blkPtrGPU(1);
    //                 len -= lpanel.nbrow(0);
    //             }
    //             panel_ptrs[mdata.batchsize] = lPanelStPtr;
    //             panel_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             panel_dim_batch[mdata.batchsize] = len;

    //             // Hackathon change: using the original diagonal block instead of the bcast buffer
    //             // diag_ptrs[mdata.batchsize] = A_gpu.dFBufs[buffer_offset];
    //             // diag_ld_batch[mdata.batchsize] = ksupc;
    //             // diag_dim_batch[mdata.batchsize] = ksupc;
    //             diag_ptrs[mdata.batchsize] = lpanel.blkPtrGPU(0);
    //             diag_ld_batch[mdata.batchsize] = lpanel.LDA();
    //             diag_dim_batch[mdata.batchsize] = ksupc;

    //             mdata.batchsize++;
    //         }
	// 	}
    // }

    // mdata.setMaxDiag();
    // mdata.setMaxPanel();

    // // Then copy the marshalled data over to the GPU
    // gpuMemcpy(mdata.dev_diag_ptrs, diag_ptrs, mdata.batchsize * sizeof(Ftype*), gpuMemcpyHostToDevice);
    // gpuMemcpy(mdata.dev_diag_ld_array, diag_ld_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice);
    // gpuMemcpy(mdata.dev_diag_dim_array, diag_dim_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice);
    // gpuMemcpy(mdata.dev_panel_ptrs, panel_ptrs, mdata.batchsize * sizeof(Ftype*), gpuMemcpyHostToDevice);
    // gpuMemcpy(mdata.dev_panel_ld_array, panel_ld_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice);
    // gpuMemcpy(mdata.dev_panel_dim_array, panel_dim_batch, mdata.batchsize * sizeof(int), gpuMemcpyHostToDevice);
}

void xLUstruct_t<Ftype>::initSCUMarshallData(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    sc_mdata.batchsize = k_end - k_st;

    MarshallInitSCUFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_maxGemmRows, sc_mdata.dev_maxGemmCols,
        sc_mdata.dev_gpu_lpanels, sc_mdata.dev_gpu_upanels, dA_gpu
    );

    gputhrust::for_each(
        gputhurst_policy, gputhrust::counting_iterator<int>(0),
        gputhrust::counting_iterator<int>(sc_mdata.batchsize), func
    );

    // for (int_t k0 = k_st; k0 < k_end; k0++)
    // {
    //     int_t k = perm_c_supno[k0];
    //     int_t buffer_offset = k0 - k_st;

    //     assert(buffer_offset < sc_mdata.upanels.size());

    //     // Wajih: TODO: figure out what this offset does
    //     int offset = 0;
    //     if (UidxSendCounts[k] > 0 && LidxSendCounts[k] > 0)
    //     {
    //         sc_mdata.upanels[buffer_offset] = getKUpanel(k, offset);
    //         sc_mdata.lpanels[buffer_offset] = getKLpanel(k, offset);

    //         // Set gemm loop parameters for the panels
    //         upanel_t& upanel = sc_mdata.upanels[buffer_offset];
    //         lpanel_t& lpanel = sc_mdata.lpanels[buffer_offset];

    //         sc_mdata.host_gpu_upanels[buffer_offset] = upanel.gpuPanel;
    //         sc_mdata.host_gpu_lpanels[buffer_offset] = lpanel.gpuPanel;

    //         if(!upanel.isEmpty() && !lpanel.isEmpty())
    //         {
    //             int_t st_lb = 0;
    //             if (myrow == krow(k))
    //                 st_lb = 1;

    //             int_t nlb = lpanel.nblocks();
    //             int_t nub = upanel.nblocks();

    //             sc_mdata.ist[buffer_offset] = st_lb;
    //             sc_mdata.iend[buffer_offset] = sc_mdata.ist[buffer_offset];

    //             int nrows = lpanel.stRow(nlb) - lpanel.stRow(st_lb);
    //             int ncols = upanel.nzcols();

    //             int maxGemmRows = nrows;
    //             int maxGemmCols = ncols;
    //             // entire gemm doesn't fit in gemm buffer
    //             if (nrows * ncols > A_gpu.gemmBufferSize)
    //             {
    //                 int maxGemmOpSize = (int)sqrt(A_gpu.gemmBufferSize);
    //                 int numberofRowChunks = (nrows + maxGemmOpSize - 1) / maxGemmOpSize;
    //                 maxGemmRows = nrows / numberofRowChunks;
    //                 maxGemmCols = A_gpu.gemmBufferSize / maxGemmRows;
    //             }

    //             sc_mdata.maxGemmRows[buffer_offset] = maxGemmRows;
    //             sc_mdata.maxGemmCols[buffer_offset] = maxGemmCols;
    //         }
    //     }
    // }

    // sc_mdata.copyPanelDataToGPU();
}

int xLUstruct_t<Ftype>::marshallSCUBatchedDataOuter(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    sc_mdata.batchsize = k_end - k_st;

    // Temporarily use the m array for the done flags
    int *done_flags = sc_mdata.dev_m_array;
    MarshallSCUOuterFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend,
        sc_mdata.dev_maxGemmRows, done_flags, dA_gpu
    );

    gputhrust::for_each(
        gputhurst_policy, gputhrust::counting_iterator<int>(0),
        gputhrust::counting_iterator<int>(sc_mdata.batchsize), func
    );

    bool done = gputhrust::all_of(
        gputhurst_policy, done_flags, done_flags + sc_mdata.batchsize,
        MarshallSCUOuter_Predicate()
    );

    return done;

    // int done_i = 1;
    // for (int k0 = k_st; k0 < k_end; k0++)
    // {
    //     int k = perm_c_supno[k0];
    //     int buffer_index = k0 - k_st;
    //     lpanel_t& lpanel = sc_mdata.lpanels[buffer_index];
    //     upanel_t& upanel = sc_mdata.upanels[buffer_index];
    //     if(lpanel.isEmpty() || upanel.isEmpty())
    //         continue;

    //     int& iEnd = sc_mdata.iend[buffer_index];
    //     // Not done if even one operation still has work to do
    //     if(iEnd < lpanel.nblocks())
    //     {
    //         done_i = 0;
    //         int& iSt = sc_mdata.ist[buffer_index];
    //         iSt = iEnd;
    //         iEnd = lpanel.getEndBlock(iSt, sc_mdata.maxGemmRows[buffer_index]);
    //         assert(iEnd > iSt);
    //         sc_mdata.jst[buffer_index] = sc_mdata.jend[buffer_index] = 0;
    //     }
    // }

    // return done_i;
}

int xLUstruct_t<Ftype>::marshallSCUBatchedDataInner(int k_st, int k_end, int_t *perm_c_supno)
{
    SCUMarshallData& sc_mdata = A_gpu.sc_marshall_data;
    int knum = k_end - k_st;
    sc_mdata.batchsize = knum;

    MarshallSCUInnerFunc func(
        k_st, sc_mdata.dev_ist, sc_mdata.dev_iend, sc_mdata.dev_jst, sc_mdata.dev_jend, sc_mdata.dev_maxGemmCols,
        sc_mdata.dev_A_ptrs, sc_mdata.dev_lda_array, sc_mdata.dev_B_ptrs, sc_mdata.dev_ldb_array, sc_mdata.dev_C_ptrs,
        sc_mdata.dev_ldc_array, sc_mdata.dev_m_array, sc_mdata.dev_n_array, sc_mdata.dev_k_array, dA_gpu
    );

    gputhrust::counting_iterator<int> start(0), end(knum);
    gputhrust::for_each(gputhurst_policy, start, end, func);

    // Set the max dims in the marshalled data
    sc_mdata.max_m = gputhrust::reduce(gputhurst_policy, sc_mdata.dev_m_array, sc_mdata.dev_m_array + knum, 0, gputhrust::maximum<int>());
    sc_mdata.max_n = gputhrust::reduce(gputhurst_policy, sc_mdata.dev_n_array, sc_mdata.dev_n_array + knum, 0, gputhrust::maximum<int>());
    sc_mdata.max_k = gputhrust::reduce(gputhurst_policy, sc_mdata.dev_k_array, sc_mdata.dev_k_array + knum, 0, gputhrust::maximum<int>());
    sc_mdata.max_ilen = gputhrust::transform_reduce(gputhurst_policy, start, end, element_diff<int>(sc_mdata.dev_ist, sc_mdata.dev_iend), 0, gputhrust::maximum<int>());
    sc_mdata.max_jlen = gputhrust::transform_reduce(gputhurst_policy, start, end, element_diff<int>(sc_mdata.dev_jst, sc_mdata.dev_jend), 0, gputhrust::maximum<int>());

    printf("SCU %d -> %d: %d %d %d %d %d\n", k_st, k_end, sc_mdata.max_m, sc_mdata.max_n, sc_mdata.max_k, sc_mdata.max_ilen, sc_mdata.max_jlen);

    return gputhrust::all_of(
        gputhurst_policy, sc_mdata.dev_m_array, sc_mdata.dev_m_array + sc_mdata.batchsize,
        MarshallSCUInner_Predicate()
    );
    // int done_j = 1;
    // for(int k0 = k_st; k0 < k_end; k0++)
    // {
    //     int k = perm_c_supno[k0];
    //     int buffer_index = k0 - k_st;
    //     lpanel_t& lpanel = sc_mdata.lpanels[buffer_index];
    //     upanel_t& upanel = sc_mdata.upanels[buffer_index];

    //     if(lpanel.isEmpty() || upanel.isEmpty())
    //         continue;

    //     int iSt = sc_mdata.ist[buffer_index];
    //     int iEnd = sc_mdata.iend[buffer_index];

    //     int& jSt = sc_mdata.jst[buffer_index];
    //     int& jEnd = sc_mdata.jend[buffer_index];

    //     // Not done if even one operation still has work to do
    //     if(jEnd < upanel.nblocks())
    //     {
    //         jSt = jEnd;
    //         jEnd = upanel.getEndBlock(jSt, sc_mdata.maxGemmCols[buffer_index]);

    //         assert(jEnd > jSt);
    //         done_j = 0;
    //         // printf("k = %d, ist = %d, iend = %d, jst = %d, jend = %d\n", k, iSt, iEnd, jSt, jEnd);

    //         sc_mdata.host_m_array[buffer_index] = lpanel.stRow(iEnd) - lpanel.stRow(iSt);
    //         sc_mdata.host_n_array[buffer_index] = upanel.stCol(jEnd) - upanel.stCol(jSt);
    //         sc_mdata.host_k_array[buffer_index] = supersize(k);

    //         sc_mdata.host_A_ptrs[buffer_index] = lpanel.blkPtrGPU(iSt);
    //         sc_mdata.host_B_ptrs[buffer_index] = upanel.blkPtrGPU(jSt);
    //         sc_mdata.host_C_ptrs[buffer_index] = A_gpu.gpuGemmBuffs[buffer_index];

    //         sc_mdata.host_lda_array[buffer_index] = lpanel.LDA();
    //         sc_mdata.host_ldb_array[buffer_index] = upanel.LDA();
    //         sc_mdata.host_ldc_array[buffer_index] = sc_mdata.host_m_array[buffer_index];
    //     }
    //     else
    //     {
    //         sc_mdata.host_A_ptrs[buffer_index] = NULL;
    //         sc_mdata.host_B_ptrs[buffer_index] = NULL;
    //         sc_mdata.host_C_ptrs[buffer_index] = NULL;

    //         sc_mdata.host_m_array[buffer_index] = 0;
    //         sc_mdata.host_n_array[buffer_index] = 0;
    //         sc_mdata.host_k_array[buffer_index] = 0;

    //         sc_mdata.host_lda_array[buffer_index] = 1;
    //         sc_mdata.host_ldb_array[buffer_index] = 1;
    //         sc_mdata.host_ldc_array[buffer_index] = 1;
    //     }
    // }

    // if(done_j == 0)
    // {
    //     // Upload the buffers to the gpu
    //     sc_mdata.setMaxDims();
    //     sc_mdata.copyToGPU();
    // }

    // return done_j;
}
#endif /* match if (0) */
