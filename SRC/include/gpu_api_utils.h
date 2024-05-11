/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 * Modified:
 *     May 22, 2022        version 8.0.0
 * </pre>
 */

#ifndef gpu_api_utils_H
#define gpu_api_utils_H

#ifdef GPU_ACC

#include "gpu_wrapper.h"
typedef struct LUstruct_gpu_  LUstruct_gpu;  // Sherry - not in this distribution

#ifndef HAVE_SYCL
#ifdef __cplusplus
extern "C" {
#endif
#endif

extern void DisplayHeader();

// AB: for now all the functions below are defined only for CUDA, HIP
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
extern const char* gpublasGetErrorString(gpublasStatus_t status);
extern gpuError_t checkGPU(gpuError_t);
extern gpublasStatus_t checkGPUblas(gpublasStatus_t);
extern gpublasHandle_t create_handle ();
extern void destroy_handle (gpublasHandle_t handle);
#endif // HAVE_CUDA, HAVE_HIP


#ifndef HAVE_SYCL
#ifdef __cplusplus
  }
#endif
#endif

#endif // end GPU_ACC
#endif // gpu_api_utils_H
