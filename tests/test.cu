
#include <cuarena/cuarena.cuh>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#define CHEADER "\x1B[1;36m"
#define CTEST "\x1B[1;34m"
#define CPASS "\x1B[1;32m"
#define CFAIL "\x1B[1;31m"

struct TestFailure : std::runtime_error {
    explicit TestFailure(const char* expr, int line)
        : std::runtime_error(std::string("line ") + std::to_string(line) + ": " + expr) 
        { }
};

#define TCHECK(expr) \
    if (!(expr)) throw TestFailure(#expr, __LINE__)

inline int total = 0, passed = 0;

inline void section(const char* title) {
    std::printf("\n%s%s%s\n", CHEADER, title, CUAR_CNORMAL);
    std::printf("%s\n", std::string(60, '-').c_str());
}

inline void gpu_cleanup() {
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess)
        cudaMemPoolTrimTo(pool, 0);
}

template<typename Func>
void run_test(const char* name, Func func) {
    ++total;
    try {
        func();
        std::printf("  %s%s%s %sPASS%s\n", 
            CTEST, name, CUAR_CNORMAL, CPASS, CUAR_CNORMAL);
        ++passed;
    } 
    catch (const TestFailure& e) {
        std::fprintf(stderr, "  %sFAIL  %s  (%s)%s\n", CFAIL, name, e.what(), CUAR_CNORMAL);
    } 
    catch (const std::exception& e) {
        std::fprintf(stderr, "  %sFAIL  %s  (unexpected exception: %s)%s\n", CFAIL, name, e.what(), CUAR_CNORMAL);
    }
    gpu_cleanup();
}

void test_gpu_device(cudaStream_t stream) {
    section("GPU Device pool");

    run_test("create / destroy", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        TCHECK(a.gpu_capacity() == 256 * cuarena::MB);
        TCHECK(a.gpu_used()     == 0);
        TCHECK(a.destroy_gpu_pool());
        TCHECK(a.gpu_capacity() == 0);
    });

    run_test("allocate / deallocate", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("available tracks capacity", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        size_t before = a.gpu_available();
        float* p = a.allocate<float>(1024);
        TCHECK(a.gpu_available() < before);
        a.deallocate(p);
        TCHECK(a.gpu_available() == before);
    });

    run_test("resize buffer", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        a.resize<float>(p, 4096);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("resize to zero deallocates", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        a.resize<float>(p, 0);
        TCHECK(p == nullptr);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("resize bail when already large enough", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        float* p    = a.allocate<float>(4096);
        float* orig = p;
        a.resize<float>(p, 1024);
        TCHECK(p == orig);
        a.deallocate(p);
    });

    run_test("free-list coalescing reclaims block", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        float* p1      = a.allocate<float>(1024);
        float* p2      = a.allocate<float>(1024);
        size_t used_two = a.gpu_used();
        a.deallocate(p1);
        a.deallocate(p2);
        TCHECK(a.gpu_used() == 0);
        float* p3 = a.allocate<float>(2048);
        TCHECK(p3 != nullptr);
        TCHECK(a.gpu_used() <= used_two);
        a.deallocate(p3);
    });

    run_test("pool exhaustion throws", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(4 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        bool threw = false;
        try { a.allocate<float>(8 * cuarena::MB); }
        catch (const cuarena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset invalidates allocations", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        a.allocate<float>(1024);
        a.allocate<float>(2048);
        a.reset_gpu_pool();
        TCHECK(a.gpu_used()      == 0);
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });

    run_test("null / zero guard", [&] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Device, stream));
        TCHECK(a.allocate<float>(0) == nullptr);
        a.deallocate<float>(nullptr);
    });
}

void test_gpu_managed() {
    section("GPU Managed pool");

    run_test("create / destroy", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        TCHECK(a.gpu_memory_type() == cuarena::GPUMemoryType::Managed);
        TCHECK(a.gpu_capacity()    == 256 * cuarena::MB);
        TCHECK(a.destroy_gpu_pool());
    });

    run_test("allocate / deallocate", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("host-accessible read/write", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        cudaDeviceSynchronize();
        int* p = a.allocate<int>(4);
        TCHECK(p != nullptr);
        p[0] = 42; p[1] = 7; p[2] = -1; p[3] = 0;
        TCHECK(p[0] == 42 && p[1] == 7 && p[2] == -1 && p[3] == 0);
        a.deallocate(p);
    });

    run_test("resize buffer", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        float* p = a.allocate<float>(1024);
        a.resize<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("free-list coalescing reclaims block", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        float* p1 = a.allocate<float>(1024);
        float* p2 = a.allocate<float>(1024);
        a.deallocate(p1);
        a.deallocate(p2);
        TCHECK(a.gpu_used() == 0);
        float* p3 = a.allocate<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(4 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        bool threw = false;
        try { a.allocate<float>(8 * cuarena::MB); }
        catch (const cuarena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuarena::MB, cuarena::GPUMemoryType::Managed));
        a.allocate<float>(1024);
        a.reset_gpu_pool();
        TCHECK(a.gpu_used()      == 0);
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });
}

void test_cpu_pinned() {
    section("CPU Pinned pool");

    run_test("create / destroy", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pinned);
        TCHECK(a.cpu_capacity()    == 128 * cuarena::MB);
        TCHECK(a.destroy_cpu_pool());
        TCHECK(a.cpu_capacity()    == 0);
    });

    run_test("allocate / deallocate", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        float* p = a.allocate_pinned<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.cpu_used() > 0);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("read/write", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        int* p = a.allocate_pinned<int>(4);
        p[0] = 1; p[1] = 2; p[2] = 3; p[3] = 4;
        TCHECK(p[0] == 1 && p[1] == 2 && p[2] == 3 && p[3] == 4);
        a.deallocate_pinned(p);
    });

    run_test("resize buffer", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        float* p = a.allocate_pinned<float>(1024);
        a.resize_pinned<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("free-list coalescing", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        float* p1 = a.allocate_pinned<float>(1024);
        float* p2 = a.allocate_pinned<float>(1024);
        a.deallocate_pinned(p1);
        a.deallocate_pinned(p2);
        TCHECK(a.cpu_used() == 0);
        float* p3 = a.allocate_pinned<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate_pinned(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(4 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        bool threw = false;
        try { a.allocate_pinned<float>(8 * cuarena::MB); }
        catch (const cuarena::cpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        a.allocate_pinned<float>(1024);
        a.reset_cpu_pool();
        TCHECK(a.cpu_used()      == 0);
        TCHECK(a.cpu_available() == a.cpu_capacity());
    });

    run_test("null / zero guard", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pinned));
        TCHECK(a.allocate_pinned<float>(0) == nullptr);
        a.deallocate_pinned<float>(nullptr);
    });
}

void test_cpu_pageable() {
    section("CPU Pageable pool");

    run_test("create / destroy", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pageable);
        TCHECK(a.cpu_capacity()    == 128 * cuarena::MB);
        TCHECK(a.destroy_cpu_pool());
        TCHECK(a.cpu_capacity()    == 0);
    });

    run_test("allocate / deallocate", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        float* p = a.allocate_pinned<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.cpu_used() > 0);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("read/write", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        int* p = a.allocate_pinned<int>(4);
        p[0] = 10; p[1] = 20; p[2] = 30; p[3] = 40;
        TCHECK(p[0] == 10 && p[1] == 20 && p[2] == 30 && p[3] == 40);
        a.deallocate_pinned(p);
    });

    run_test("resize buffer", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        float* p = a.allocate_pinned<float>(1024);
        a.resize_pinned<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("free-list coalescing", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        float* p1 = a.allocate_pinned<float>(1024);
        float* p2 = a.allocate_pinned<float>(1024);
        a.deallocate_pinned(p1);
        a.deallocate_pinned(p2);
        TCHECK(a.cpu_used() == 0);
        float* p3 = a.allocate_pinned<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate_pinned(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(4 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        bool threw = false;
        try { a.allocate_pinned<float>(8 * cuarena::MB); }
        catch (const cuarena::cpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuarena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuarena::MB, cuarena::CPUMemoryType::Pageable));
        a.allocate_pinned<float>(1024);
        a.reset_cpu_pool();
        TCHECK(a.cpu_used()      == 0);
        TCHECK(a.cpu_available() == a.cpu_capacity());
    });
}

void test_constructor_combinations(cudaStream_t stream) {
    section("Constructor combinations");

    run_test("Device + Pinned", [&] {
        cuarena::DeviceArena a(128 * cuarena::MB, 256 * cuarena::MB,
                               cuarena::GPUMemoryType::Device,
                               cuarena::CPUMemoryType::Pinned, stream);
        TCHECK(a.gpu_memory_type() == cuarena::GPUMemoryType::Device);
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Device + Pageable", [&] {
        cuarena::DeviceArena a(128 * cuarena::MB, 256 * cuarena::MB,
                               cuarena::GPUMemoryType::Device,
                               cuarena::CPUMemoryType::Pageable, stream);
        TCHECK(a.gpu_memory_type() == cuarena::GPUMemoryType::Device);
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Managed + Pinned", [] {
        cuarena::DeviceArena a(128 * cuarena::MB, 256 * cuarena::MB,
                               cuarena::GPUMemoryType::Managed,
                               cuarena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_memory_type() == cuarena::GPUMemoryType::Managed);
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Managed + Pageable", [] {
        cuarena::DeviceArena a(128 * cuarena::MB, 256 * cuarena::MB,
                               cuarena::GPUMemoryType::Managed,
                               cuarena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_memory_type() == cuarena::GPUMemoryType::Managed);
        TCHECK(a.cpu_memory_type() == cuarena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });
}

int main() {
    cuarena::Logger::set_level(0);

    cudaStream_t stream;
    CUARENA_CHECK(cudaStreamCreate(&stream));

    test_gpu_device(stream);
    test_gpu_managed();
    test_cpu_pinned();
    test_cpu_pageable();
    test_constructor_combinations(stream);

    CUARENA_CHECK(cudaStreamDestroy(stream));

    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("  %s%d%s / %d tests passed%s\n", passed == total ? CPASS : CFAIL, passed, CHEADER, total, CUAR_CNORMAL);
    std::printf("%s\n\n", std::string(60, '=').c_str());

    return (passed == total) ? 0 : 1;
}
