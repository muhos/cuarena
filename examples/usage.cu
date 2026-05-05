#include <cuarena/cuarena.cuh>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

/**
 * Persistent buffers: stable region, allocated once.
 * Scratch buffers: dynamic region, frequently created and destroyed.
 */

int main() {
    cuArena::Logger::set_level(1);

    try {
        cudaStream_t stream;
        CUARENA_CHECK(cudaStreamCreate(&stream));

        // Pool layout:
        //   total:   4 GB
        //   stable:  512 MB
        //   dynamic: 3584 MB
        cuArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * cuArena::GB, cuArena::GPUMemoryType::Device, stream, 512 * cuArena::MB);
        alloc.create_cpu_pool(512 * cuArena::MB, cuArena::CPUMemoryType::Pinned);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        std::cout << std::format("\n  Pool created with 512 MB stable and {} MB dynamic\n",
            (alloc.gpu_capacity() - 512 * cuArena::MB) / cuArena::MB);

        float* p1 = alloc.allocate<float>(64  * cuArena::MB / sizeof(float), cuArena::Region::Stable);
        float* p2 = alloc.allocate<float>(128 * cuArena::MB / sizeof(float), cuArena::Region::Stable);
        float* p3 = alloc.allocate<float>(256 * cuArena::MB / sizeof(float), cuArena::Region::Stable);
        CUARENA_CHECK(cudaMemsetAsync(p1, 0, 64  * cuArena::MB, stream));
        CUARENA_CHECK(cudaMemsetAsync(p2, 0, 128 * cuArena::MB, stream));
        CUARENA_CHECK(cudaMemsetAsync(p3, 0, 256 * cuArena::MB, stream));
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        std::cout << std::format("  Persistent buffers allocated, stable used: {} MB / 512 MB\n",
            alloc.gpu_stable_used() / cuArena::MB);

        float* h_stage = alloc.allocate_pinned<float>(32 * cuArena::MB / sizeof(float));
        std::cout << std::format("  Pinned mirror: {} MB\n",
            alloc.cpu_used() / cuArena::MB);

        // Fill most of the dynamic pool with equal-sized blocks, 
        // then free every other one.
        constexpr size_t BLOCK = 128 * cuArena::MB / sizeof(float); 
        constexpr int    N     = 24;

        std::vector<float*> dynamics;
        dynamics.reserve(N);

        std::cout << "\n  Filling dynamic region with " << N << " x 128 MB blocks...\n";
        for (int i = 0; i < N; i++)
            dynamics.push_back(alloc.allocate<float>(BLOCK));
        std::cout << "  Freeing every other block to create fragmentation...\n";
        for (int i = 0; i < N; i += 2) {
            alloc.deallocate(dynamics[i]);
            dynamics[i] = nullptr;
        }

        size_t avail   = alloc.gpu_available();
        size_t used    = alloc.gpu_used() - alloc.gpu_stable_used();
        size_t largest = alloc.gpu_largest_free_block();
        std::cout << std::format("  Before compact:  dynamic used {:4d} MB  available {:4d} MB  largest block {:4d} MB\n",
            used / cuArena::MB, avail / cuArena::MB, largest / cuArena::MB);

        if (largest < avail / 2) {
            std::cout << "  Fragmentation detected, compacting dynamic region...";

            std::unordered_map<cuArena::addr_t, cuArena::addr_t> map;
            alloc.compact_gpu_dynamic(
                [&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
                    map[old_p] = new_p;
                },
            stream);
            CUARENA_CHECK(cudaStreamSynchronize(stream));

            for (auto& p : dynamics) {
                if (!p) continue;
                auto it = map.find(p);
                if (it != map.end()) p = static_cast<float*>(it->second);
            }
            std::cout << std::format(" done (largest block: {} MB -> {} MB)\n",
                largest / cuArena::MB, alloc.gpu_largest_free_block() / cuArena::MB);
        }

        for (auto p : dynamics) if (p) alloc.deallocate(p);
        dynamics.clear();

        std::cout << std::format("\n  After all iterations, dynamic used: {} MB\n",
            (alloc.gpu_used() - alloc.gpu_stable_used()) / cuArena::MB);

        alloc.deallocate(p1);
        alloc.deallocate(p2);
        alloc.deallocate(p3);
        alloc.deallocate_pinned(h_stage);

        std::cout << std::format("  Total GPU used after clean-up: {} MB\n",
            alloc.gpu_used() / cuArena::MB);

        alloc.destroy_gpu_pool();
        alloc.destroy_cpu_pool();
        CUARENA_CHECK(cudaStreamDestroy(stream));

    }
    catch (const cuArena::gpu_memory_error& e) {
        std::cerr << std::format("GPU memory error: {}\n", e.what());
        return 1;
    }
    catch (const cuArena::cpu_memory_error& e) {
        std::cerr << std::format("CPU memory error: {}\n", e.what());
        return 1;
    }
    catch (const cuArena::cuda_error& e) {
        std::cerr << std::format("CUDA error: {}\n", e.what());
        return 1;
    }

    return 0;
}
