#pragma once

#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "types.hpp"
#include "logger.hpp"

namespace cuarena {

struct cuda_error : std::runtime_error {
    explicit cuda_error(const char* msg) : std::runtime_error(msg) {}
};

#ifdef __GNUC__
#define CUARENA_FORCEINLINE __attribute__((always_inline)) inline
#else
#define CUARENA_FORCEINLINE __forceinline
#endif

#define CUARENA_DEVICE     __device__
#define CUARENA_HOST       __host__
#define CUARENA_HOSTDEVICE __host__ __device__

#define CUARENA_CHECK(call)                                                          \
    do {                                                                            \
        const cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                                    \
            cuarena::Logger::error("CUDA error [%s:%d]: %s",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));                        \
            throw cuarena::cuda_error(cudaGetErrorString(_e));                      \
        }                                                                           \
    } while (0)

#define CUARENA_LASTERR(msg)                                                         \
    do {                                                                            \
        const cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                                    \
            cuarena::Logger::error("%s [%s:%d]: %s",                                \
                msg, __FILE__, __LINE__, cudaGetErrorString(_e));                   \
            throw cuarena::cuda_error(cudaGetErrorString(_e));                      \
        }                                                                           \
    } while (0)

#define CUARENA_SYNC(stream)  CUARENA_CHECK(cudaStreamSynchronize(stream))
#define CUARENA_SYNCALL       CUARENA_CHECK(cudaDeviceSynchronize())

#ifdef CUARENA_SYNC_ALLOC
#define CUARENA_MALLOC(pptr, sz, stream) cudaMalloc((pptr), (sz))
#define CUARENA_FREE(ptr, stream)        cudaFree((ptr))
#else
#define CUARENA_MALLOC(pptr, sz, stream) cudaMallocAsync((pptr), (sz), (stream))
#define CUARENA_FREE(ptr, stream)        cudaFreeAsync((ptr), (stream))
#endif

}
