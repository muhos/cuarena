#include <cuarena/cuarena.cuh>
#include <format>
#include <iostream>

int main() {
    cuarena::Logger::set_level(1);

    try {
        cuarena::DeviceArena alloc(512 * cuarena::MB, 2 * cuarena::GB);

        float* d_a = alloc.allocate<float>(1 << 20);
        CUARENA_CHECK(cudaMemset(d_a, 0, (1 << 20) * sizeof(float)));

        int*   d_b = alloc.allocate<int>(256);
        float* d_c = alloc.allocate<float>(1 << 18);

        std::cout << "After 3 allocations:\n";
        std::cout << std::format("  GPU used:      {} MB\n", alloc.gpu_used()      / cuarena::MB);
        std::cout << std::format("  GPU available: {} MB\n", alloc.gpu_available() / cuarena::MB);

        alloc.deallocate(d_b);
        alloc.deallocate(d_c);

        std::cout << "After freeing d_b and d_c (coalesced back into free list):\n";
        std::cout << std::format("  GPU used:      {} MB\n", alloc.gpu_used()      / cuarena::MB);
        std::cout << std::format("  GPU available: {} MB\n", alloc.gpu_available() / cuarena::MB);

        alloc.resize<float>(d_a, 1 << 21);

        std::cout << "After resize d_a x2:\n";
        std::cout << std::format("  GPU used:      {} MB\n", alloc.gpu_used()      / cuarena::MB);
        std::cout << std::format("  GPU available: {} MB\n", alloc.gpu_available() / cuarena::MB);

        float* h_buf = alloc.allocate_pinned<float>(1 << 10);
        std::cout << std::format("CPU pinned used:      {} KB\n", alloc.cpu_used()      / cuarena::KB);
        std::cout << std::format("CPU pinned available: {} MB\n", alloc.cpu_available() / cuarena::MB);

        alloc.deallocate(d_a);
        alloc.deallocate_pinned(h_buf);

        alloc.reset_gpu_pool();
        std::cout << "After GPU reset:\n";
        std::cout << std::format("  GPU available: {} MB\n", alloc.gpu_available() / cuarena::MB);

    } 
    catch (const cuarena::gpu_memory_error& e) {
        std::cerr << std::format("GPU memory error: {}\n", e.what());
        return 1;
    } 
    catch (const cuarena::cpu_memory_error& e) {
        std::cerr << std::format("CPU memory error: {}\n", e.what());
        return 1;
    } 
    catch (const cuarena::cuda_error& e) {
        std::cerr << std::format("CUDA error: {}\n", e.what());
        return 1;
    }

    return 0;
}
