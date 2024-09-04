#pragma once
#include "gpu_wrapper.h"

template <typename Ftype>
gpusolverStatus_t myGpusolverGetrf(gpusolverDnHandle_t handle, int m, int n, Ftype *A, int lda, Ftype *Workspace, int *devIpiv, int *devInfo);

template <>
gpusolverStatus_t myGpusolverGetrf<double>(gpusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo)
{
    return gpusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
gpusolverStatus_t myGpusolverGetrf<float>(gpusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo)
{
    return gpusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
gpusolverStatus_t myGpusolverGetrf<gpuComplex>(gpusolverDnHandle_t handle, int m, int n, gpuComplex *A, int lda, gpuComplex *Workspace, int *devIpiv, int *devInfo)
{
    return gpusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
gpusolverStatus_t myGpusolverGetrf<gpuDoubleComplex>(gpusolverDnHandle_t handle, int m, int n, gpuDoubleComplex *A, int lda, gpuDoubleComplex *Workspace, int *devIpiv, int *devInfo)
{
    return gpusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <typename Ftype>
gpublasStatus_t myGpublasTrsm(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag, int m, int n, const Ftype *alpha, const Ftype *A, int lda, Ftype *B, int ldb);

template <>
gpublasStatus_t myGpublasTrsm<double>(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)
{
    return gpublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myGpublasTrsm<float>(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)
{
    return gpublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myGpublasTrsm<gpuComplex>(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag, int m, int n, const gpuComplex *alpha, const gpuComplex *A, int lda, gpuComplex *B, int ldb)
{
    return gpublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
gpublasStatus_t myGpublasTrsm<gpuDoubleComplex>(gpublasHandle_t handle, gpublasSideMode_t side, gpublasFillMode_t uplo, gpublasOperation_t trans, gpublasDiagType_t diag, int m, int n, const gpuDoubleComplex *alpha, const gpuDoubleComplex *A, int lda, gpuDoubleComplex *B, int ldb)
{
    return gpublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename Ftype>
gpublasStatus_t myGpublasScal(gpublasHandle_t handle, int n, const Ftype *alpha, Ftype *x, int incx);

template <typename Ftype>
gpublasStatus_t myGpublasAxpy(gpublasHandle_t handle, int n, const Ftype *alpha, const Ftype *x, int incx, Ftype *y, int incy);

template <>
gpublasStatus_t myGpublasScal<double>(gpublasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    return gpublasDscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myGpublasScal<float>(gpublasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    return gpublasSscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myGpublasAxpy<double>(gpublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
    return gpublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myGpublasAxpy<float>(gpublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
{
    return gpublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myGpublasScal<gpuComplex>(gpublasHandle_t handle, int n, const gpuComplex *alpha, gpuComplex *x, int incx)
{
    return gpublasCscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myGpublasScal<gpuDoubleComplex>(gpublasHandle_t handle, int n, const gpuDoubleComplex *alpha, gpuDoubleComplex *x, int incx)
{
    return gpublasZscal(handle, n, alpha, x, incx);
}

template <>
gpublasStatus_t myGpublasAxpy<gpuComplex>(gpublasHandle_t handle, int n, const gpuComplex *alpha, const gpuComplex *x, int incx, gpuComplex *y, int incy)
{
    return gpublasCaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
gpublasStatus_t myGpublasAxpy<gpuDoubleComplex>(gpublasHandle_t handle, int n, const gpuDoubleComplex *alpha, const gpuDoubleComplex *x, int incx, gpuDoubleComplex *y, int incy)
{
    return gpublasZaxpy(handle, n, alpha, x, incx, y, incy);
}

template <typename Ftype>
gpublasStatus_t myGpublasGemm(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const Ftype *alpha, const Ftype *A, int lda, const Ftype *B, int ldb, const Ftype *beta, Ftype *C, int ldc);

template <>
gpublasStatus_t myGpublasGemm<double>(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    return gpublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myGpublasGemm<float>(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    return gpublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myGpublasGemm<gpuComplex>(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const gpuComplex *alpha, const gpuComplex *A, int lda, const gpuComplex *B, int ldb, const gpuComplex *beta, gpuComplex *C, int ldc)
{
    return gpublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myGpublasGemm<gpuDoubleComplex>(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const gpuDoubleComplex *alpha, const gpuDoubleComplex *A, int lda, const gpuDoubleComplex *B, int ldb, const gpuDoubleComplex *beta, gpuDoubleComplex *C, int ldc)
{
    return gpublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
gpublasStatus_t myGpublasGemm<doublecomplex>(gpublasHandle_t handle, gpublasOperation_t transa, gpublasOperation_t transb, int m, int n, int k, const doublecomplex *alpha, const doublecomplex *A, int lda, const doublecomplex *B, int ldb, const doublecomplex *beta, doublecomplex *C, int ldc)
{
    // return gpublasZgemm(handle, transa, transb, m, n, k, 
    // alpha, A, lda, B, ldb, beta, C, ldc);
    // cast doublecomplex to gpuDoubleComplex
    return gpublasZgemm(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const gpuDoubleComplex *>(alpha),
        reinterpret_cast<const gpuDoubleComplex *>(A), lda,
        reinterpret_cast<const gpuDoubleComplex *>(B), ldb,
        reinterpret_cast<const gpuDoubleComplex *>(beta),
        reinterpret_cast<gpuDoubleComplex *>(C), ldc);
    
}

template <>
cusolverStatus_t myGpusolverGetrf<doublecomplex>(
    cusolverDnHandle_t handle, int m, int n, doublecomplex *A, int lda,
    doublecomplex *Workspace, int *devIpiv, int *devInfo)
{
    // return gpusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
    // cast doublecomplex to gpuDoubleComplex
    return gpusolverDnZgetrf(
        handle, m, n, reinterpret_cast<gpuDoubleComplex *>(A), lda,
        reinterpret_cast<gpuDoubleComplex *>(Workspace), devIpiv, devInfo);
}

// now creating the wrappers for the other functions 
template <>
gpublasStatus_t myGpublasTrsm<doublecomplex>(gpublasHandle_t handle,
                                           gpublasSideMode_t side, gpublasFillMode_t uplo,
                                           gpublasOperation_t trans, gpublasDiagType_t diag,
                                           int m, int n,
                                           const doublecomplex *alpha,
                                           const doublecomplex *A, int lda,
                                           doublecomplex *B, int ldb) {
    // Your implementation here
    // You can use cublasZtrsm function because it's for gpuDoubleComplex type
    return gpublasZtrsm(handle, side, uplo, trans, diag, m, n, 
                       reinterpret_cast<const gpuDoubleComplex*>(alpha), 
                       reinterpret_cast<const gpuDoubleComplex*>(A), lda, 
                       reinterpret_cast<gpuDoubleComplex*>(B), ldb);
}

template <>
gpublasStatus_t myGpublasScal<doublecomplex>(gpublasHandle_t handle, int n, 
                                           const doublecomplex *alpha, 
                                           doublecomplex *x, int incx) {
    // Your implementation here
    // You can use cublasZscal function because it's for gpuDoubleComplex type
    return gpublasZscal(handle, n, reinterpret_cast<const gpuDoubleComplex*>(alpha), 
                       reinterpret_cast<gpuDoubleComplex*>(x), incx);
}

template <>
gpublasStatus_t myGpublasAxpy<doublecomplex>(gpublasHandle_t handle, int n, 
                                           const doublecomplex *alpha, 
                                           const doublecomplex *x, int incx, 
                                           doublecomplex *y, int incy) {
    // Your implementation here
    // You can use cublasZaxpy function because it's for gpuDoubleComplex type
    return gpublasZaxpy(handle, n, reinterpret_cast<const gpuDoubleComplex*>(alpha), 
                       reinterpret_cast<const gpuDoubleComplex*>(x), incx, 
                       reinterpret_cast<gpuDoubleComplex*>(y), incy);
}


// gpublasStatus_t myGpublasScal<doublecomplex> 
// gpublasStatus_t myGpublasAxpy<doublecomplex>
// gpublasStatus_t myGpublasGemm<doublecomplex>
