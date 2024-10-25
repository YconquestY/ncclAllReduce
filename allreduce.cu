/*
 * This file tests NCCL allreduce between 2 nodes.
 *
 * Copyright (c) 2024 by Yue Yu
 */
#include <nccl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <random>
#include <vector>
#include <thread>

constexpr size_t cnt = 1024 * 1024 * 1024 / sizeof(half);
constexpr int    dSize = 8;
constexpr float  low = 0.f,
                 high = 1.f;
constexpr int    seed = 595;

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

template<typename S, typename T>
void random_fill(S* v, T low, T high)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(static_cast<float>(low),
                                              static_cast<float>(high));
    for (size_t i = 0; i < cnt; ++i) {
        v[i] = static_cast<S>(dis(gen));
    }
}

void threadFn(int rank, ncclUniqueId* id)
{
    CUDACHECK(cudaSetDevice(rank));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, dSize, *id, rank));

    half* hIn  = new half[cnt]; random_fill(hIn, low, high);
    half* hOut = new half[cnt];

    half* dIn;
    half* dOut;
    CUDACHECK(cudaMalloc(&dIn , cnt * sizeof(half)));
    CUDACHECK(cudaMalloc(&dOut, cnt * sizeof(half)));

    CUDACHECK(cudaMemcpyAsync(dIn, hIn, cnt * sizeof(half), cudaMemcpyHostToDevice, stream));
    NCCLCHECK(ncclAllReduce(
        reinterpret_cast<const void*>(dIn),
        reinterpret_cast<void*>(dOut),
        cnt,
        ncclHalf,
        ncclSum,
        comm,
        stream
    ));
    CUDACHECK(cudaMemcpyAsync(hOut, dOut, cnt * sizeof(half), cudaMemcpyDeviceToHost, stream));

    NCCLCHECK(ncclCommDestroy(comm));
    delete[] hIn;
    delete[] hOut;
    CUDACHECK(cudaFree(dIn));
    CUDACHECK(cudaFree(dOut));
}

int main(int argc, char* argv[])
{
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    
    std::vector<std::thread> workers; workers.reserve(dSize);
    for (int i = 0; i < dSize; ++i) {
        workers.emplace_back(
            threadFn,
            i,
            &id
        );
    }

    for (auto&& t : workers) {
        t.join();
    }
}