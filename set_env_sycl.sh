module use /soft/modulefiles/
module load oneapi/upstream
module load spack-pe-base cmake

#export CUDA_VISIBLE_DEVICES=3
export SUPERLU_ACC_OFFLOAD=1
export SUPERLU_NUM_GPU_STREAMS=1
export OMP_NUM_THREADS=1
