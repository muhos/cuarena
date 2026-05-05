![GitHub License](https://img.shields.io/github/license/muhos/cuArena)
[![Build Status](https://github.com/muhos/cuArena/actions/workflows/test-build.yml/badge.svg)](https://github.com/muhos/cuArena/actions/workflows/test-build.yml)

# cuArena

Pool-based CUDA memory allocator with isolated-bin allocation, address-ordered coalescing free lists, stable/dynamic memory regions, and on-demand defragmentation. I built cuArena for my own CUDA projects. All my allocators will be migrating to it as their memory backend. If it fits your use case, you're welcome to use it too.

---

## Overview

cuArena pre-allocates a contiguous block of GPU memory and a block of CPU memory at startup, then services all subsequent allocations entirely in host-side C++ with no `cudaMalloc` call is made per allocation.

The GPU pool is divided into two regions:

- **Stable region** (front of pool), for long-term allocations that persist across iterations. Uses a linear pointer with coalescing-only free list. Compact never touches it.
- **Dynamic region** (rest of pool), for temporary allocations that are frequently created and destroyed. Uses isolated size-class bins for $O(1)$ allocation in the common case, with coalescing on every free. Supports on-demand compaction to eliminate fragmentation.

The design is suited for CUDA applications that perform many repeated allocations of varying sizes in performance-critical paths, where the overhead of `cudaMalloc` and `cudaFree` becomes measurable. AI frameworks like PyTorch and TensorFlow ship their own pool allocators for this reason but those allocators are inseparable from their runtimes. cuArena offers a similar approach as a lightweight, embeddable library for any CUDA application.

---

## Designed for Performance

- **Zero driver overhead**, every allocation and deallocation is a pure host-side operation; no `cudaMalloc` per call
- **$O(1)$ allocation**, 32 isolated size-class bins covering $[128\,\text{B},\,256\,\text{GB})$; the right bin is found in one step, no scanning
- **Immediate coalescing**, adjacent free blocks merge on every `deallocate()`, keeping the free list compact without a separate GC pass
- **On-demand defragmentation**, `compact_gpu_dynamic()` eliminates fragmentation in one async pass when needed; a callback delivers the updated pointers

---

## Requirements

| Dependency | Minimum version |
|---|---|
| CUDA Toolkit | 12.x |
| C++ standard | C++20 |
| CMake | 3.18 |
| GCC | 13 (for `std::format` in examples) |

---

## Build

```bash
git clone https://github.com/muhos/cuArena.git
cd cuArena
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the usage example:
```bash
./build/cuArena_usage
```

Run the benchmark:
```bash
./build/cuArena_benchmark
```

Run the tests:
```bash
./build/cuArena_test
```

### Options

| Option | Default | Description |
|---|---|---|
| `CUARENA_BUILD_EXAMPLES` | `ON` | Build `cuArena_usage` and `cuArena_benchmark` |
| `CUARENA_BUILD_TESTS` | `ON` | Build `cuArena_test` |
| `CUARENA_SYNC_ALLOC` | `OFF` | Use `cudaMalloc` instead of `cudaMallocAsync` |

Example:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUARENA_SYNC_ALLOC=ON
```

---

## Integration

After building, `build/` contains the static library (`libcuarena.a`) and headers are under `include/cuarena/`. Point your build system at both and link against `libcuarena.a` and the CUDA runtime. Alternatively, install into a system prefix:

```bash
cmake --install build --prefix /usr/local
```

---

## Quick start

```cpp
#include <cuarena/cuarena.cuh>

int main() {
    cuArena::Logger::set_level(1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuArena::DeviceArena alloc;

    // 4 GB GPU pool: 512 MB stable for long-term buffers, rest dynamic for temporaries
    alloc.create_gpu_pool(4 * cuArena::GB, cuArena::GPUMemoryType::Device, stream, 512 * cuArena::MB);
    alloc.create_cpu_pool(512 * cuArena::MB, cuArena::CPUMemoryType::Pinned);
    cudaStreamSynchronize(stream);

    // Stable allocation which remains alive for the duration of the session
    float* persistent = alloc.allocate<float>(128 * cuArena::MB / sizeof(float), cuArena::Region::Stable);

    // Dynamic allocations that frequently created and destroyed
    float* scratch = alloc.allocate<float>(256 * cuArena::MB / sizeof(float));
    alloc.deallocate(scratch);

    // Compact dynamic region when fragmentation builds up
    std::unordered_map<cuArena::addr_t, cuArena::addr_t> map;
    alloc.compact_gpu_dynamic(
        [&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            map[old_p] = new_p;
        }, 
    stream);
    cudaStreamSynchronize(stream);

    // Pinned CPU buffer for host <-> device transfers
    float* h_stage = alloc.allocate_pinned<float>(32 * cuArena::MB / sizeof(float));

    alloc.deallocate(persistent);
    alloc.deallocate_pinned(h_stage);
    alloc.destroy_gpu_pool();
    alloc.destroy_cpu_pool();
    cudaStreamDestroy(stream);
}
```

---

## API

### Types

```cpp
enum class GPUMemoryType { Device, Managed };
enum class CPUMemoryType { Pinned, Pageable };
enum class Region        { Stable, Dynamic };

// Callback fired by compact_gpu_dynamic for each relocated block.
// Runs after all cudaMemcpyAsync calls are queued; sync the stream before
// accessing data through the new pointers.
using RelocateCallback = std::function<void(addr_t old_ptr, addr_t new_ptr, size_t bytes)>;
```

### Pool lifecycle

```cpp
// Construct and create both pools immediately
DeviceArena(size_t cpu_limit, size_t gpu_limit,
            GPUMemoryType gtype  = GPUMemoryType::Device,
            CPUMemoryType ctype  = CPUMemoryType::Pinned,
            cudaStream_t  stream = 0);

// Create pools manually (limit = 0 uses all available memory minus a safety margin)
// stable_bytes: size of the stable region at the front of the GPU pool (0 = no stable region)
bool create_gpu_pool(size_t limit        = 0,
                     GPUMemoryType type  = GPUMemoryType::Device,
                     cudaStream_t stream = 0,
                     size_t stable_bytes = 0);
bool create_cpu_pool(size_t limit        = 0,
                     CPUMemoryType type  = CPUMemoryType::Pinned);

// Destroy pools and release memory to the driver
bool destroy_gpu_pool();
bool destroy_cpu_pool();

// Resize (invalidates all existing allocations, preserves type and stable proportion)
bool resize_gpu_pool(size_t new_size, cudaStream_t stream = 0);
bool resize_cpu_pool(size_t new_size);

// Reset free lists (invalidates all existing allocations, no GPU free/malloc)
void reset_gpu_pool();
void reset_cpu_pool();
```

### GPU allocation

```cpp
// region defaults to Dynamic; pass Region::Stable for long-term allocations
template<class T> T*   allocate  (size_t count, Region region = Region::Dynamic);
template<class T> void deallocate(T* ptr);
template<class T> void resize    (T*& ptr, size_t new_count);  // Dynamic only
```

### Compaction

```cpp
// Compact all dynamic allocations to the front of the dynamic region.
// Queues cudaMemcpyAsync calls on `stream`; caller must sync before using new pointers.
void compact_gpu_dynamic(RelocateCallback callback, cudaStream_t stream = 0);
```

### CPU pinned allocation

```cpp
template<class T> T*   allocate_pinned  (size_t count);
template<class T> void deallocate_pinned(T* ptr);
template<class T> void resize_pinned    (T*& ptr, size_t new_count);
```

### Stats

```cpp
addr_t gpu_base             ();  // raw base pointer of the GPU pool
size_t gpu_capacity         ();  // total GPU pool size
size_t gpu_stable_capacity  ();  // size of the stable region
size_t gpu_dynamic_capacity ();  // size of the dynamic region
size_t gpu_used             ();  // bytes used across both regions
size_t gpu_stable_used      ();  // bytes used in the stable region
size_t gpu_available        ();  // free bytes in the dynamic region

size_t cpu_capacity ();
size_t cpu_used     ();
size_t cpu_available();
```

### Logger

```cpp
cuArena::Logger::set_level(0);  // silent (default)
cuArena::Logger::set_level(1);  // info: pool creation, destruction, resize
cuArena::Logger::set_level(2);  // debug
```

### Error handling

```cpp
try {
    float* p = alloc.allocate<float>(too_many);
}
catch (const cuArena::gpu_memory_error& e) { ... }
catch (const cuArena::cpu_memory_error& e) { ... }
catch (const cuArena::cuda_error& e)       { ... }
```

---

## Benchmark results

Measured on RTX 4090, CUDA 12.6, Ubuntu 24.04, GCC 13.3. 500 iterations, 50 warmup. All times in microseconds.

### Device pool

**Benchmark 1: single large alloc + free (8 MB)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMalloc` + `cudaFree` | 603.72 | 595.62 | 591.79 | 620.20 |
| `cudaMallocAsync` + `cudaFreeAsync` | 1.38 | 0.28 | 0.27 | 550.73 |
| cuArena alloc + dealloc | **0.06** | **0.06** | **0.05** | **0.08** |

cuArena is **$10137{\times}$ faster** than `cudaMalloc` and **$23{\times}$ faster** than `cudaMallocAsync`.

**Benchmark 2: batch alloc 512 $\times$ 1 KB, then free all**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMalloc` + `cudaFree` | 849.91 | 854.49 | 833.00 | 941.48 |
| `cudaMallocAsync` + `cudaFreeAsync` | 165.09 | 163.34 | 160.58 | 842.58 |
| cuArena alloc + dealloc | **47.86** | **47.74** | **47.42** | **51.84** |

cuArena is **$17.76{\times}$ faster** than `cudaMalloc` and **$3.45{\times}$ faster** than `cudaMallocAsync`.

**Benchmark 3: interleaved ping-pong (4 live $\times$ 1 MB, cycling)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMalloc` + `cudaFree` | 0.39 | 0.33 | 0.30 | 0.63 |
| `cudaMallocAsync` + `cudaFreeAsync` | 0.29 | 0.29 | 0.27 | 0.50 |
| cuArena alloc + dealloc | **0.06** | **0.06** | **0.06** | **0.08** |

cuArena is **$6.25{\times}$ faster** than `cudaMalloc` and **$4.70{\times}$ faster** than `cudaMallocAsync`.

**Benchmark 4: resize chain (1 MB to 8 MB in 8 doublings)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMalloc` + `cudaFree` | 7183.49 | 7180.44 | 7174.56 | 7301.32 |
| `cudaMallocAsync` + `cudaFreeAsync` | 4.19 | 2.88 | 2.83 | 642.14 |
| cuArena resize | **0.66** | **0.65** | **0.64** | **1.31** |

cuArena is **$10893{\times}$ faster** than `cudaMalloc` and **$6.35{\times}$ faster** than `cudaMallocAsync`.

**Benchmark 5: fragmentation + compact cycle (8 $\times$ 32 MB, free alternating)**

| operation | mean | median | min | max |
|---|---|---|---|---|
| `compact_gpu_dynamic` (256 MB live) | 296.37 | 296.25 | 293.98 | 299.67 |
| alloc 64 MB after compact | **0.05** | **0.05** | **0.04** | **0.32** |

Compact + subsequent alloc total: **296.43 $\mu s$**. The compact cost is purely GPU memcpy bandwidth where 128 MB of data is transferred. After compaction, the 64 MB allocation that would have failed before completes in 0.05 $\mu s$.

**Benchmark 6: stable vs dynamic allocation latency**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| stable 256 B | **0.05** | **0.05** | **0.05** | **0.06** |
| dynamic 256 B | 0.06 | 0.06 | 0.06 | 0.08 |
| stable 256 KB | **0.06** | **0.06** | **0.06** | **0.07** |
| dynamic 256 KB | 0.09 | 0.09 | 0.09 | 0.10 |
| stable 2 MB | **0.07** | **0.06** | **0.06** | **0.13** |
| dynamic 2 MB | 0.09 | 0.09 | 0.09 | 0.10 |

Stable is $1.24–1.48{\times}$ faster than dynamic across all size classes. The gap reflects the bin-lookup overhead in the dynamic allocator vs. the stable region's simple linear pointer.

### Managed pool

**Benchmark 1: single large alloc + free (8 MB)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMallocManaged` + `cudaFree` | 16.48 | 16.36 | 15.62 | 20.69 |
| cuArena alloc + dealloc | **0.06** | **0.06** | **0.06** | **0.07** |

cuArena is **$270{\times}$ faster** than `cudaMallocManaged`.

**Benchmark 2: batch alloc 512 $\times$ 1 KB, then free all**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMallocManaged` + `cudaFree` | 2254.53 | 2248.98 | 2224.77 | 2446.00 |
| cuArena alloc + dealloc | **46.58** | **46.11** | **45.33** | **55.13** |

cuArena is **$48{\times}$ faster** than `cudaMallocManaged`.

**Benchmark 3: interleaved ping-pong (4 live $\times$ 1 MB, cycling)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMallocManaged` + `cudaFree` | 4.29 | 4.35 | 3.88 | 8.12 |
| cuArena alloc + dealloc | **0.07** | **0.08** | **0.06** | **0.09** |

cuArena is **$60{\times}$ faster** than `cudaMallocManaged`.

**Benchmark 4: resize chain (1 MB to 8 MB in 8 doublings)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `cudaMallocManaged` + `cudaFree` | 136.80 | 124.94 | 122.00 | 1321.96 |
| cuArena resize | **0.64** | **0.63** | **0.62** | **3.54** |

cuArena is **$214{\times}$ faster** than `cudaMallocManaged`.

### Remarks

`cudaMallocAsync` shows extreme max latency spikes (up to $842 \mu s$) against a median of $0.27–163 \mu s$ caused by unpredictable driver-side stream pool overhead. cuArena's max values stay within $2\times$ of its median across all benchmarks, reflecting the deterministic nature of pure host-side allocation.

`cudaMallocManaged` carries full UVM page table setup per call. The batch benchmark shows $2254 \mu s$ vs $849 \mu s$ for `cudaMalloc`, and cuArena's managed pool amortizes that entire cost to a single call at creation time. The suballocations run at 0.06–0.09 $\mu s$ regardless.

The resize benchmark captures compounding cost most clearly: 8 sequential `cudaMalloc` calls accumulate to $7.1 ms$ per chain, while cuArena's free-list reuse keeps the entire chain under $1 \mu s$ for both pool types.

Compact throughput (~432 GB/s on RTX 4090) shows that `compact_gpu_dynamic` adds essentially no overhead beyond the raw GPU memcpy bandwidth required to move the data.
