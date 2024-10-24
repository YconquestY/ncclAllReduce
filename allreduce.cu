/*
 * This file tests NCCL allreduce between 2 nodes.
 *
 * Copyright (c) 2024 by Yue Yu
 */
#include <mpi.h>

#include <nccl.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

constexpr size_t cnt = 1;

#define MPICHECK(cmd) do {  \
  int e = cmd;              \
  if (e != MPI_SUCCESS) {   \
    std::printf("MPI error %s: %d '%d'\n",  __FILE__,__LINE__, e); \
    exit(EXIT_FAILURE);     \
  }                         \
} while(0)

#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd;      \
  if (e != cudaSuccess) {   \
    std::printf("CUDA error %s: %d '%s'\n", __FILE__,__LINE__, cudaGetErrorString(e));\
    exit(EXIT_FAILURE);     \
  }                         \
} while(0)


#define NCCLCHECK(cmd) do { \
  ncclResult_t r = cmd;     \
  if (r != ncclSuccess) {   \
    std::printf("NCCL error %s: %d '%s'\n", __FILE__,__LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE);     \
  }                         \
} while(0)

int main(int argc, char* argv[])
{
    int hRank,
        hSize;

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &hRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &hSize));

    ncclUniqueId id;
    if (hRank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast((void*) &id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, hSize, id, hRank)); // Each node provides 1 device.
    CUDACHECK(cudaSetDevice(0));

    float* dIn;
    float* dOut;
    CUDACHECK(cudaMalloc(&dIn , cnt * sizeof(float)));
    CUDACHECK(cudaMalloc(&dOut, cnt * sizeof(float)));

    float* hIn  = new float[cnt];
    for (int i = 0; i < cnt; ++i) {
        hIn[i] = static_cast<float>(i + 1);
    }
    float* hOut = new float[cnt];

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    CUDACHECK(cudaMemcpyAsync(dIn, hIn, cnt * sizeof(float), cudaMemcpyHostToDevice, stream));
    NCCLCHECK(ncclAllReduce(
        reinterpret_cast<const void*>(dIn),
        reinterpret_cast<void*>(dOut),
        cnt,
        ncclFloat,
        ncclSum,
        comm,
        stream
    ));

    CUDACHECK(cudaMemcpyAsync(hOut, dOut, cnt * sizeof(float), cudaMemcpyDeviceToHost, stream));

    std::printf("rank %d: ", hRank);
    for (int i = 0; i < cnt; ++i) {
        std::printf("%f ", hOut[i]);
    }
    std::printf("\n");
    
    MPICHECK(MPI_Finalize());
    delete[] hIn;
    delete[] hOut;
    CUDACHECK(cudaFree(dIn));
    CUDACHECK(cudaFree(dOut));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaStreamDestroy(stream));
}