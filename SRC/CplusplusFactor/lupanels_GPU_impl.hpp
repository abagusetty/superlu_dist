#pragma once
#include <cassert>
#include <algorithm>
#include <cmath>
#include "superlu_defs.h"
#include "superlu_dist_config.h"

// #ifdef HAVE_CUDA
#define EPSILON 1e-3

#include "lupanels.hpp"

#include <cmath>
#include <complex>
#include <cassert>

template<typename T>
int checkArr(const T *A, const T *B, int n)
{
    double nrmA = 0;
    for (int i = 0; i < n; i++) {
        // For complex numbers, std::norm gives the squared magnitude.
        nrmA += sqnorm(A[i]);
    }
    nrmA = std::sqrt(nrmA);

    for (int i = 0; i < n; i++) {
        // Use std::abs for both real and complex numbers to get the magnitude.
        // assert(std::abs(A[i] - B[i]) <= EPSILON * nrmA / n);
        assert(std::sqrt(sqnorm(A[i] - B[i])) <= EPSILON * nrmA / n);
    }

    return 0;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    checkGPU(gpuMalloc((void**)&gpuPanel.index, idxSize));
    checkGPU(gpuMalloc((void**)&gpuPanel.val, valSize));

    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
xlpanelGPU_t<T> xlpanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (T *) basePtr;

    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int_t xlpanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    checkGPU(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int_t xupanel_t<T>::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    checkGPU(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
    return 0;
}

template <typename T>
int xupanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    checkGPU(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

template <typename T>
int xlpanel_t<T>::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(T) * nzvalSize();
    checkGPU(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU()
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    checkGPU(gpuMalloc((void**)&gpuPanel.index, idxSize));
    checkGPU(gpuMalloc((void**)&gpuPanel.val, valSize));

    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));
    return gpuPanel;
}

template <typename T>
xupanelGPU_t<T> xupanel_t<T>::copyToGPU(void* basePtr)
{
    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(T) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (T *) basePtr;

    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

template <typename T>
int xlpanel_t<T>::checkGPUPanel()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    checkGPU(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                                  int_t ksupsz,
                                  T *DiagBlk, // device pointer
                                  int_t LDD)
{
    if (isEmpty())
        return 0;
    T *lPanelStPtr = blkPtrGPU(0); // &val[blkPtrOffset(0)];
    int_t len = nzrows();
    if (haveDiag())
    {
        lPanelStPtr = blkPtrGPU(1); // &val[blkPtrOffset(1)];
        len -= nbrow(0);
    }

    T alpha = one<T>();

    #ifdef HAVE_SYCL
    if constexpr (std::is_same_v<T, doublecomplex>) {
      oneapi::mkl::blas::column_major::trsm(*cuStream,
                                            GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                                            GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                                            len, ksupsz, gpuDoubleComplex(alpha.r, alpha.i), reinterpret_cast<const gpuDoubleComplex*>(DiagBlk), LDD,
                                            reinterpret_cast<gpuDoubleComplex*>(lPanelStPtr), LDA());
    } else {
      oneapi::mkl::blas::column_major::trsm(*cuStream,
                                            GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                                            GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                                            len, ksupsz, alpha, DiagBlk, LDD,
                                            lPanelStPtr, LDA());
    }
    #else
    gpublasSetStream(handle, cuStream);
    gpublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, &alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA());
    #endif

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::diagFactorPackDiagBlockGPU(int_t k,
                                           T *UBlk, int_t LDU,     // CPU pointers
                                           T *DiagLBlk, int_t LDD, // CPU pointers
                                           T thresh, int_t *xsup,
                                           superlu_dist_options_t *options,
                                           SuperLUStat_t *stat, int *info)
{
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    checkGPU(gpuMemcpy2D(DiagLBlk, dpitch, val, spitch,
                 width, height, gpuMemcpyDeviceToHost));

    // call dgetrf2
    dgstrf2(k, DiagLBlk, LDD, UBlk, LDU,
            thresh, xsup, options, stat, info);

    //copy back to device
    checkGPU(gpuMemcpy2D(val, spitch, DiagLBlk, dpitch,
                 width, height, gpuMemcpyHostToDevice));

    return 0;
}

template <typename T>
int_t xlpanel_t<T>::diagFactorGpuSolver(int_t k,
                                        gpusolverDnHandle_t cusolverH, gpuStream_t cuStream,
                                        T *dWork, int* dInfo,  // GPU pointers
                                        T *dDiagBuf, int_t LDD, // GPU pointers
                                        threshPivValType<T> thresh, int_t *xsup,
                                        superlu_dist_options_t *options,
                                        SuperLUStat_t *stat, int *info)
{
    // gpuStream_t stream = NULL;
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(T);
    size_t spitch = LDA() * sizeof(T);
    size_t width = kSupSize * sizeof(T);
    size_t height = kSupSize;
    T *val = blkPtrGPU(0);

    #ifdef HAVE_SYCL
    if constexpr (std::is_same_v<T, doublecomplex>) {
      auto scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<gpuDoubleComplex>(*cuStream, kSupSize, kSupSize, LDA());
      oneapi::mkl::lapack::getrf(*cuStream, kSupSize, kSupSize, reinterpret_cast<gpuDoubleComplex*>(val), LDA(), NULL,  reinterpret_cast<gpuDoubleComplex*>(dWork), scratchpad_size);
    } else {
      std::cout << "1. value of custream: " << cuStream << std::endl;
      auto scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<T>(*cuStream, kSupSize, kSupSize, LDA());
      std::cout << "2. value of custream: " << cuStream << std::endl;
      oneapi::mkl::lapack::getrf(*cuStream, kSupSize, kSupSize, val, LDA(), NULL, dWork, scratchpad_size);
      std::cout << "3. value of custream: " << cuStream << std::endl;
    }
    #else
    gpusolverCheckErrors(gpusolverDnSetStream(cusolverH, cuStream));
    gpusolverCheckErrors(myCusolverGetrf<T>(cusolverH, kSupSize, kSupSize, val, LDA(), dWork, NULL, dInfo));
    #endif

    checkGPU(gpuMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                 width, height, gpuMemcpyDeviceToDevice, cuStream));
    checkGPU(gpuStreamSynchronize(cuStream));
    return 0;
}

template <typename T>
int_t xupanel_t<T>::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                                  int_t ksupsz,
                                  T *DiagBlk,
                                  int_t LDD)
{
    if (isEmpty())
        return 0;

    T alpha = one<T>();

    #ifdef HAVE_SYCL
    if constexpr (std::is_same_v<T, doublecomplex>) {
      oneapi::mkl::blas::column_major::trsm(*cuStream,
                                            GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                                            GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                                            ksupsz, nzcols(), gpuDoubleComplex(alpha.r, alpha.i), reinterpret_cast<const gpuDoubleComplex*>(DiagBlk), LDD,
                                            reinterpret_cast<gpuDoubleComplex*>(blkPtrGPU(0)), LDA());
    } else {
      oneapi::mkl::blas::column_major::trsm(*cuStream,
                                            GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                                            GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                                            ksupsz, nzcols(), alpha, DiagBlk, LDD,
                                            blkPtrGPU(0), LDA());
    }
    #else
    gpublasSetStream(handle, cuStream);
    gpublasStatus_t cbstatus =
        myCublasTrsm<T>(handle,
                    GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                    ksupsz, nzcols(), &alpha, DiagBlk, LDD,
                    blkPtrGPU(0), LDA());
    #endif

    return 0;
}

template <typename T>
int xupanel_t<T>::checkGPUPanel()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(T) * nzvalSize();

    std::vector<T> tmpArr(nzvalSize());
    checkGPU(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}
