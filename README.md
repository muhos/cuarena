# cuarena

Pool-based CUDA memory allocator with $O(log\,n)$ best-fit allocation, address-ordered coalescing free lists, and zero driver overhead. I built cuarena for my own CUDA projects. All my allocators will be migrating to it as their memory backend. If it fits your use case, you're welcome to use it too.

## Overview

cuarena pre-allocates a contiguous block of GPU device memory and a block of pinned CPU memory at startup, then services all subsequent allocations entirely in host-side C++ &ndash; no `cudaMalloc` call is made per allocation. Freed blocks are returned to a free list and immediately coalesced with adjacent free neighbors, eliminating fragmentation over time without moving data around.

The design is suited for CUDA applications that perform many repeated allocations of varying sizes in performance-critical paths, where the overhead of `cudaMalloc` and `cudaFree` becomes measurable. AI frameworks like PyTorch and TensorFlow ship their own pool allocators for this reason but those allocators are inseparable from their runtimes. cuarena does similar approach but as a lightweight, embeddable library for any CUDA application.

## Designed for Performance

- **Zero driver overhead per allocation**, allocation and deallocation are pure host-side operations after pool creation
- **Best-fit with splitting**, $O(log\,n)$ allocation from a size-indexed free list; oversized free blocks are split to avoid waste
- **Immediate coalescing**, on every `deallocate()`, adjacent free blocks are merged in $O(log\,n)$ using an address-indexed map; no deferred compaction pass required
- **Dual pool**, independent GPU device pool (`cudaMallocAsync`) and CPU pinned pool (`cudaMallocHost`) with symmetric APIs
- **Thread-safe**, all (de)allocations are protected by a mutex
- **Stream-aware pool creation**, GPU pool creation and resize accept a GPU stream for concurrency
- **Typed-template API** , \{`allocate<T>`, `deallocate<T>`, `resize<T>`\} with automatic size and alignment that matches GPU L1 cache line

## Requirements

| Dependency | Minimum version |
|---|---|
| CUDA Toolkit | 12.x or later |
| C++ standard | C++20 |
| CMake | 3.18 |
| GCC | 13 (for `std::format` in examples) |

## Build

```bash
git clone https://github.com/yourname/cuarena.git
cd cuarena
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the usage example:
```bash
./build/cuarena_basic
```

Run the benchmark:
```bash
./build/cuarena_benchmark
```

### Build options

| Option | Default | Description |
|---|---|---|
| `CUARENA_BUILD_EXAMPLES` | `ON` | Build `cuarena_basic` and `cuarena_benchmark` |
| `CUARENA_SYNC_ALLOC` | `OFF` | Use `cudaMalloc` instead of `cudaMallocAsync` |

Example:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUARENA_SYNC_ALLOC=ON
```

## Integration

After building, the build/ directory contains the static library (`libcuarena.a`) and the headers are under `include/cuarena/`. Point your build system at both and link against `libcuarena.a` and the CUDA runtime.
Alternatively, run `cmake --install` to place the headers and library into a system prefix:
```bash
cmake --install build --prefix /usr/local
```

## Quick start

```cpp
#include <cuarena/cuarena.cuh>

int main() {
    cuarena::Logger::set_level(1);

    // Create pools: 1 GB pinned CPU, 4 GB GPU
    cuarena::DeviceArena alloc(1 * cuarena::GB, 4 * cuarena::GB);

    // GPU allocation
    float* d_buf = alloc.allocate<float>(1 << 20);

    // Pinned CPU allocation (support concurrent transfers)
    float* h_buf = alloc.allocate_pinned<float>(1 << 20);

    // Grow a buffer without copying (existing data is discarded)
    alloc.resize<float>(d_buf, 1 << 22);

    // Freed blocks return to the free list and coalesce with neighbors
    alloc.deallocate(d_buf);
    alloc.deallocate_pinned(h_buf);

    // Reset all live allocations at once (constant time)
    alloc.reset_gpu_pool();
}
```

## API

### Pool lifecycle

```cpp
// Construct and create both pools immediately
DeviceArena(size_t cpu_limit, size_t gpu_limit, cudaStream_t stream = 0);

// Create pools manually (limit = 0 uses all available memory minus a safety penalty)
bool create_gpu_pool(size_t limit = 0, cudaStream_t stream = 0);
bool create_cpu_pool(size_t limit = 0);

// Destroy pools and release memory back to the OS
bool destroy_gpu_pool();
bool destroy_cpu_pool();

// Resize pools (invalidates all existing allocations)
bool resize_gpu_pool(size_t new_size, cudaStream_t stream = 0);
bool resize_cpu_pool(size_t new_size);

// Reset pools and free lists (invalidates all existing allocations)
void reset_gpu_pool();
void reset_cpu_pool();
```

### GPU allocation

```cpp
template<class T> T*   allocate(const size_t& count);
template<class T> void deallocate(T* ptr);
template<class T> void resize(T*& ptr, const size_t& new_count);
```

### CPU pinned allocation

```cpp
template<class T> T*   allocate_pinned(size_t count);
template<class T> void deallocate_pinned(T* ptr);
template<class T> void resize_pinned(T*& ptr, size_t new_count);
```

### Stats

```cpp
size_t gpu_capacity();   // total pool size
size_t gpu_used();       // bytes currently allocated
size_t gpu_available();  // free pool memory

size_t cpu_capacity();
size_t cpu_used();
size_t cpu_available();
```

### Logger

```cpp
cuarena::Logger::set_level(0);  // silent (default)
cuarena::Logger::set_level(1);  // informative: pool creation, destruction, resize
cuarena::Logger::set_level(2);  // debug
```

### Error handling

All allocation failures throw typed exceptions catchable independently:

```cpp
try {
    float* p = alloc.allocate<float>(too_many);
} 
catch (const cuarena::gpu_memory_error& e) { ... }
catch (const cuarena::cpu_memory_error& e) { ... }
catch (const cuarena::cuda_error& e)       { ... }
```
