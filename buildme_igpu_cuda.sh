#!/bin/bash

NOW=$(date +"%m-%d-%Y")
installdir=$PWD/install_igpu
echo $installdir

bdir=build_cuda-$NOW

rm -rf make.inc $bdir; mkdir $bdir; cd $bdir
rm -rf ${installdir}
#export PARMETIS_ROOT=/gpfs/jlse-fs0/users/dguo/tests/Parmetis/parmetis-4.0.3

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DTPL_ENABLE_PARMETISLIB=FALSE \
      -DTPL_ENABLE_LAPACKLIB=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DTPL_ENABLE_CUDALIB=TRUE \
      -Denable_openmp:BOOL=FALSE \
      -Denable_complex16:BOOL=FALSE \
      -DTPL_ENABLE_COMBBLASLIB=OFF \
      -DTPL_ENABLE_INTERNAL_BLASLIB=ON \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_STATIC_LIBS=ON \
      -DCMAKE_C_COMPILER=cc \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_Fortran_COMPILER=ftn \
      -DCMAKE_CUDA_FLAGS="-I${CRAY_MPICH_DIR}/include" \
      -DCMAKE_C_FLAGS="-DPRNTlevel=1 -DPI_DEBUG=1 -DDEBUGlevel=0" \
      -DCMAKE_CXX_FLAGS="-DPRNTlevel=1 -DPI_DEBUG=1 -DDEBUGlevel=0" \
      -DCMAKE_INSTALL_PREFIX=${installdir} \
      -DXSDK_INDEX_SIZE=64 \
      -DXSDK_ENABLE_Fortran=OFF \
      ..
