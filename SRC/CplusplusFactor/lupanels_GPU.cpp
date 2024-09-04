#include <cassert>
#include <algorithm>
#include <cmath>
#include "superlu_defs.h"
#include "superlu_dist_config.h"

#ifdef GPU_ACC

#include "lupanels.hpp"

//TODO: make expsilon a enviroment variable
// #define EPSILON 1e-3
#define EPSILON 1e-6

#if 0
// int checkArr(double *A, double *B, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         assert(fabs(A[i] - B[i]) <= EPSILON * std::min(fabs(A[i]), fabs(B[i])));
//     }

//     return 0;
// }
#else
int checkArr(double *A, double *B, int n)
{
    double nrmA = 0;
    for (int i = 0; i < n; i++)
        nrmA += A[i]*A[i];
    nrmA = sqrt(nrmA);
    for (int i = 0; i < n; i++)
    {
        assert(fabs(A[i] - B[i]) <= EPSILON * nrmA/n );
    }

    return 0;
}
#endif
lpanelGPU_t lpanel_t::copyToGPU()
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();


    checkGPU(gpuMalloc((void**)&gpuPanel.index, idxSize));
    checkGPU(gpuMalloc((void**)&gpuPanel.val, valSize));


    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

lpanelGPU_t lpanel_t::copyToGPU(void* basePtr)
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    // gpuMalloc(&gpuPanel.index, idxSize);
    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (double *) basePtr;
    // gpuMalloc(&gpuPanel.val, valSize);

    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}

int_t lpanel_t::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    checkGPU(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
}

int_t upanel_t::copyFromGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    checkGPU(gpuMemcpy(val, gpuPanel.val,  valSize, gpuMemcpyDeviceToHost));
}

int upanel_t::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    checkGPU(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

int lpanel_t::copyBackToGPU()
{
    if(isEmpty())
        return 0;
    size_t valSize = sizeof(double) * nzvalSize();
    checkGPU(gpuMemcpy(gpuPanel.val, val,  valSize, gpuMemcpyHostToDevice));
}

upanelGPU_t upanel_t::copyToGPU()
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    checkGPU(gpuMalloc((void**)&gpuPanel.index, idxSize));
    checkGPU(gpuMalloc((void**)&gpuPanel.val, valSize));


    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));
    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));
    return gpuPanel;
}

upanelGPU_t upanel_t::copyToGPU(void* basePtr)
{

    if (isEmpty())
        return gpuPanel;
    size_t idxSize = sizeof(int_t) * indexSize();
    size_t valSize = sizeof(double) * nzvalSize();

    gpuPanel.index = (int_t*) basePtr;
    // gpuMalloc(&gpuPanel.index, idxSize);
    checkGPU(gpuMemcpy(gpuPanel.index, index, idxSize, gpuMemcpyHostToDevice));

    basePtr = (char *)basePtr+ idxSize;
    gpuPanel.val = (double *) basePtr;
    // gpuMalloc(&gpuPanel.val, valSize);

    checkGPU(gpuMemcpy(gpuPanel.val, val, valSize, gpuMemcpyHostToDevice));

    return gpuPanel;
}


int lpanel_t::checkGPUPanel()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(double) * nzvalSize();

    std::vector<double> tmpArr(nzvalSize());
    checkGPU(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());

    return 0;
}

int_t lpanel_t::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz,
                              double *DiagBlk, // device pointer
                              int_t LDD)
{
    if (isEmpty())
        return 0;
    double *lPanelStPtr = blkPtrGPU(0); // &val[blkPtrOffset(0)];
    int_t len = nzrows();
    if (haveDiag())
    {
        /* code */
        lPanelStPtr = blkPtrGPU(1); // &val[blkPtrOffset(1)];
        len -= nbrow(0);
    }

    double alpha = 1.0;

    #ifdef HAVE_SYCL
    oneapi::mkl::blas::column_major::trsm(*cuStream,
                    GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA());
    #else
    gpublasCheckErrors( gpublasSetStream(handle, cuStream) );
    gpublasCheckErrors( gpublasDtrsm(handle,
                    GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                    GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
                    len, ksupsz, &alpha, DiagBlk, LDD,
                    lPanelStPtr, LDA()) );
    #endif

    return 0;
}

int_t lpanel_t::diagFactorPackDiagBlockGPU(int_t k,
                                           double *UBlk, int_t LDU,     // CPU pointers
                                           double *DiagLBlk, int_t LDD, // CPU pointers
                                           double thresh, int_t *xsup,
                                           superlu_dist_options_t *options,
                                           SuperLUStat_t *stat, int *info)
{
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;
    double *val = blkPtrGPU(0);

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

int_t lpanel_t::diagFactorGpuSolver(int_t k,
                                    gpusolverDnHandle_t gpusolverH, gpuStream_t cuStream,
                                    double *dWork, int* dInfo,  // GPU pointers
                                    double *dDiagBuf, int_t LDD, // GPU pointers
                                    double thresh, int_t *xsup,
                                    superlu_dist_options_t *options,
                                    SuperLUStat_t *stat, int *info)
{
    // gpusolverDnHandle_t gpusolverH = NULL;
    gpuStream_t stream = NULL;
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;
    double *val = blkPtrGPU(0);
    // gpusolverDnDgetrf_bufferSize(gpusolverH, m, m, d_A, lda, &lwork)

    // call the gpusolver
    #ifndef HAVE_SYCL
    gpusolverCheckErrors(gpusolverDnSetStream(gpusolverH, cuStream));
    gpusolverCheckErrors(gpusolverDnDgetrf(gpusolverH, kSupSize, kSupSize, val, LDA(), dWork, NULL, dInfo));
    #else
    auto scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(*cuStream, kSupSize, kSupSize, LDA());
    oneapi::mkl::lapack::getrf(*cuStream, kSupSize, kSupSize, val, LDA(), nullptr, dWork, scratchpad_size);
    #endif

    // Device to Device Copy
    checkGPU(gpuMemcpy2DAsync(dDiagBuf, dpitch, val, spitch,
                 width, height, gpuMemcpyDeviceToDevice, cuStream));
    checkGPU(gpuStreamSynchronize(cuStream));
    return 0;
}

int_t upanel_t::panelSolveGPU(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{
    if (isEmpty())
        return 0;

    double alpha = 1.0;

    #ifdef HAVE_SYCL
    gpublasCheckErrors(oneapi::mkl::blas::column_major::trsm(*cuStream,
                                     GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                                     GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                                     ksupsz, nzcols(), alpha, DiagBlk, LDD,
                                     blkPtrGPU(0), LDA()));
    #else
    gpublasCheckErrors( gpublasSetStream(handle, cuStream) );
    gpublasCheckErrors( gpublasDtrsm(handle,
                                     GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                                     GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
                                     ksupsz, nzcols(), &alpha, DiagBlk, LDD,
                                     blkPtrGPU(0), LDA()) );
    #endif
}

int upanel_t::checkGPUPanel()
{
    assert(isEmpty() == gpuPanel.isEmpty());

    if (isEmpty())
        return 0;

    size_t valSize = sizeof(double) * nzvalSize();

    // double *tmpArr = new double[nzvalSize()];
    std::vector<double> tmpArr(nzvalSize());
    checkGPU(gpuMemcpy(tmpArr.data(), gpuPanel.val, valSize, gpuMemcpyDeviceToHost));

    int out = checkArr(tmpArr.data(), val, nzvalSize());
    // delete tmpArr;

    return 0;
}

#if 0
lpanelGPU_t::lpanelGPU_t(lpanel_t &lpanel) : lpanel_CPU(lpanel)
{
    size_t idxSize = sizeof(int_t) * lpanel.indexSize();
    size_t valSize = sizeof(double) * lpanel.nzvalSize();


    cudaMalloc(&index, idxSize);
    cudaMemcpy(index, lpanel.index, idxSize, cudaMemcpyHostToDevice);

    cudaMalloc(&val, valSize);
    cudaMemcpy(val, lpanel.val, valSize, cudaMemcpyHostToDevice);
}

int lpanelGPU_t::check(lpanel_t &lpanel)
{
    // both should be simulatnously empty or non empty
    assert(isEmpty() == lpanel.isEmpty());

    size_t valSize = sizeof(double) * lpanel.nzvalSize();

    double *tmpArr = double[lpanel.nzvalSize()];
    cudaMemcpy(tmpArr, val, valSize, cudaMemcpyDeviceToHost);

    int out = checkArr(tmpArr, lpanel.val, lpanel.nzvalSize());
    delete tmpArr;
    return 0;
}

int_t lpanelGPU_t::panelSolve(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{

    if (lpanel_CPU.isEmpty())
        return 0;
    double *lPanelStPtr = &val[lpanel_CPU.blkPtrOffset(0)];
    int_t len = lpanel_CPU.nzrows();
    if (lpanel_CPU.haveDiag())
    {
        /* code */
        lPanelStPtr = &val[lpanel_CPU.blkPtrOffset(1)];
        len -= lpanel_CPU.nbrow(0);
    }

    double alpha = 1.0;

    gpublasStatus_t cbstatus = gpublasDtrsm(handle,
                                          GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER,
                                          𝖢𝖴BLAS_OP_𝖭, GPUBLAS_DIAG_NON_UNIT,
                                          len, ksupsz, alpha, DiagBlk, LDD,
                                          lPanelStPtr, lpanel_CPU.LDA());

    // if (isEmpty()) return 0;
    // double *lPanelStPtr = blkPtr(0);
    // int_t len = nzrows();
    // if (haveDiag())
    // {
    //     /* code */
    //     lPanelStPtr = blkPtr(1);
    //     len -= nbrow(0);
    // }
    // double alpha = 1.0;
    // superlu_dtrsm("R", "U", "N", "N",
    //               len, ksupsz, alpha, DiagBlk, LDD,
    //               lPanelStPtr, LDA());
}

int_t lpanelGPU_t::diagFactorPackDiagBlock(int_t k,
                                           double *UBlk, int_t LDU,
                                           double *DiagLBlk, int_t LDD,
                                           double thresh, int_t *xsup,
                                           superlu_dist_options_t *options, SuperLUStat_t *stat, int *info)
{
    // pack and transfer to CPU
    // cudaMemcpy2D
    int kSupSize = SuperSize(k);
    size_t dpitch = LDD * sizeof(double);
    size_t spitch = lpanel_CPU.LDA() * sizeof(double);
    size_t width = kSupSize * sizeof(double);
    size_t height = kSupSize;

    cudaMemcpy2D(DiagLBlk, dpitch, val, spitch,
                 width, height, cudaMemcpyDeviceToHost);

    // call dgetrf2
    dgstrf2(k, DiagLBlk, LDD, UBlk, LDU,
            thresh, xsup, options, stat, info);

    //copy back to device
    cudaMemcpy2D(val, spitch, DiagLBlk, dpitch,
                 width, height, cudaMemcpyHostToDevice);

    return 0;
}


upanelGPU_t::upanelGPU_t(upanel_t &upanel) : upanel_CPU(upanel)
{
    size_t idxSize = sizeof(int_t) * upanel.indexSize();
    size_t valSize = sizeof(double) * upanel.nzvalSize();

    cudaMalloc(&index, idxSize);
    cudaMemcpy(index, upanel.index, idxSize, cudaMemcpyHostToDevice);

    cudaMalloc(&val, valSize);
    cudaMemcpy(val, upanel.val, valSize, cudaMemcpyHostToDevice);
}


int upanelGPU_t::check(upanel_t &upanel)
{
    // both should be simulatnously empty or non empty
    assert(isEmpty() == upanel.isEmpty());

    size_t valSize = sizeof(double) * upanel.nzvalSize();

    double *tmpArr = double[upanel.nzvalSize()];
    cudaMemcpy(tmpArr, val, valSize, cudaMemcpyDeviceToHost);

    int out = checkArr(tmpArr, upanel.val, upanel.nzvalSize());
    delete tmpArr;
    return 0;
}



int_t upanelGPU_t::panelSolve(gpublasHandle_t handle, gpuStream_t cuStream,
                              int_t ksupsz, double *DiagBlk, int_t LDD)
{
    if (upanel_CPU.isEmpty())
        return 0;

    double alpha = 1.0;

    gpublasStatus_t cbstatus =
        gpublasDtrsm(handle,
                    𝖢𝖴BLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER,
                    𝖢𝖴BLAS_OP_𝖭, GPUBLAS_DIAG_UNIT,
                    ksupsz, upanel_CPU.nzcols(), alpha, DiagBlk, LDD,
                    val, upanel_CPU.LDA());

    // superlu_dtrsm("L", "L", "N", "U",
    //               ksupsz, nzcols(), 1.0, DiagBlk, LDD, val, LDA());
    return 0;
}
#endif // if 0
#endif // GPU_ACC
