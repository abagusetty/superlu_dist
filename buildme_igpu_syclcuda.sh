#!/bin/bash

NOW=$(date +"%m-%d-%Y")

bdir=build_sycl-${NOW}
installdir=$PWD/install_igpu-${NOW}
echo $installdir


rm -rf make.inc $bdir; mkdir $bdir; cd $bdir
rm -rf ${installdir}
#export PARMETIS_ROOT=/gpfs/jlse-fs0/users/dguo/tests/Parmetis/parmetis-4.0.3

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DTPL_ENABLE_PARMETISLIB=FALSE \
      -DTPL_ENABLE_LAPACKLIB=OFF \
      -DTPL_ENABLE_SYCLLIB=TRUE \
      -Denable_single=ON \
      -Denable_openmp:BOOL=FALSE \
      -DTPL_ENABLE_COMBBLASLIB=OFF \
      -DTPL_ENABLE_INTERNAL_BLASLIB=ON \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_Fortran_COMPILER=gfortran \
      -DCMAKE_C_FLAGS="-DPRNTlevel=1 -DDEBUGlevel=0 -Wall" \
      -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -L${MKLROOT}/lib -lonemkl -DPRNTlevel=1 -DDEBUGlevel=0 -Wall" \
      -DCMAKE_INSTALL_PREFIX=${installdir} \
      -DXSDK_INDEX_SIZE=32 \
      -DXSDK_ENABLE_Fortran=OFF \
      -DHAVE_MAGMA=NO \
      ..

#-DPI_DEBUG=1
#make pddrive -j8


#export CUBLAS_LOGINFO_DBG=1
#export CUBLAS_LOGDEST_DBG=stdout
