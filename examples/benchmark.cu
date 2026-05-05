#include <cuarena/cuarena.cuh>
#include <chrono>
#include <vector>
#include <format>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

#define SEP_SIZE 84

#define CSECTION  "\x1B[1;34m"
#define CBENCH    "\x1B[1;36m"
#define CFASTER   "\x1B[1;32m" 
#define CSLOWER   "\x1B[1;31m"

using Clock = std::chrono::high_resolution_clock;

struct Stats {
    double mean;
    double median;
    double min;
    double max;
};

struct Config {
    static constexpr size_t ITERS  = 500;
    static constexpr size_t WARMUP = 50;
    static constexpr size_t SIZE_SMALL  = 256;
    static constexpr size_t SIZE_MEDIUM = 256 * cuArena::KB;
    static constexpr size_t SIZE_LARGE  = 2   * cuArena::MB;
    static constexpr size_t BATCH_SIZE = 512;
};

inline double elapsed(const Clock::time_point& start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

inline Stats compute_stats(std::vector<double> times) {
    if (times.empty()) return Stats{0, 0, 0, 0};
    Stats s;
    std::sort(times.begin(), times.end());
    s.min    = times.front();
    s.max    = times.back();
    s.median = times[times.size() / 2];
    double sum  = 0;
    for (const auto& t : times) sum += t;
    s.mean   = sum / times.size();
    return s;
}

inline void print_separator(const char& sep = '-') {
    std::cout << std::format("{}\n", std::string(SEP_SIZE, sep));
}

inline void print_header(const char* title) {
    std::cout << std::format(CBENCH "{}" CUAR_CNORMAL "\n", title);
    print_separator('=');
    std::cout << std::format("{:<34}  {:>10}  {:>11}  {:>10}  {:>10}\n",
        "allocator", "mean (us)", "median (us)", "min (us)", "max (us)");
    print_separator();
}

inline void print_row(const char* label, const Stats& s) {
    std::cout << std::format("{:<34}  {:10.2f}  {:11.2f}  {:10.2f}  {:10.2f}\n",
        label, s.mean, s.median, s.min, s.max);
}

inline void print_speedup(const char* fast, double fast_mean,
                           const char* slow, double slow_mean) {
    const bool faster = fast_mean < slow_mean;
    const double ratio = faster ? slow_mean / fast_mean : fast_mean / slow_mean;
    std::cout << std::format("cuArena vs {:<22}  {}{:.2f}x {}" CUAR_CNORMAL "\n",
        slow, faster ? CFASTER : CSLOWER, ratio, faster ? "faster" : "slower");
}

inline void print_footer(const Stats& cuda_sync, const Stats& cuda_async, const Stats& arena, const bool& is_device = true) {
    print_separator();
    print_speedup("cuArena", arena.mean, is_device ? "cudaMalloc (sync)" : "cudaMallocManaged",  cuda_sync.mean);
    if (is_device) print_speedup("cuArena", arena.mean, "cudaMallocAsync",    cuda_async.mean);
    print_separator('=');
    std::cout << '\n';
}

inline void print_stats(const std::vector<double>& t_sync, 
                        const std::vector<double>& t_async, 
                        const std::vector<double>& t_arena, 
                        const char* memType, const char* areType, 
                        const bool& is_device) {
    auto s_sync  = compute_stats(t_sync);
    auto s_async = compute_stats(t_async);
    auto s_arena = compute_stats(t_arena);
    print_row(memType, s_sync);
    if (is_device) print_row("cudaMallocAsync + cudaFreeAsync", s_async);
    print_row(areType, s_arena);
    print_footer(s_sync, s_async, s_arena, is_device);
}

void run_benchmark(cuArena::DeviceArena& alloc, const cudaStream_t& stream = 0);

int main() {
    cuArena::Logger::set_level(0);

    // cuArena benchmark (device pool)
    {
        cudaStream_t stream;
        CUARENA_CHECK(cudaStreamCreate(&stream));
        cuArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * cuArena::GB, cuArena::GPUMemoryType::Device, stream);
        CUARENA_CHECK(cudaStreamSynchronize(stream));
        std::cout << std::format(CSECTION "cuArena benchmark (device)  —  {} iterations, {} warmup:" CUAR_CNORMAL "\n\n", Config::ITERS, Config::WARMUP);
        run_benchmark(alloc, stream);
        alloc.destroy_gpu_pool();
        CUARENA_CHECK(cudaStreamDestroy(stream));
    }
    std::cout << std::endl;
    // cuArena benchmark (managed pool)
    {
        cuArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * cuArena::GB, cuArena::GPUMemoryType::Managed);
        CUARENA_CHECK(cudaDeviceSynchronize());
        std::cout << std::format(CSECTION "cuArena benchmark (managed)  —  {} iterations, {} warmup:" CUAR_CNORMAL "\n\n", Config::ITERS, Config::WARMUP);
        run_benchmark(alloc);
    }
    
    return 0;
}

void run_benchmark(cuArena::DeviceArena& alloc, const cudaStream_t& stream) {
    const bool is_device = alloc.gpu_memory_type() == cuArena::GPUMemoryType::Device;
    using AllocFunc = std::function<cudaError_t(void**, size_t)>;
    AllocFunc allocfunc = is_device ? AllocFunc([ &stream ](void** p, size_t s) { return CUARENA_MALLOC(p, s, stream); })
                                    : AllocFunc([ ]        (void** p, size_t s) { return cudaMallocManaged(p, s); });
    std::string memType = is_device ? "cudaMalloc      " : "cudaMallocManaged  ";
    memType += "+ cudaFree";
    std::string areType = is_device ? "cuArena alloc   " : "cuArena alloc      ";
    areType += "+ dealloc";
    // --------------------------------------
    // Benchmark 1: single large alloc + free
    // --------------------------------------
    print_header("Benchmark 1: single large alloc + free  (8 MB per iteration)");
    {
        float* p = nullptr;
        // warmup with sync single
        for (size_t i = 0; i < Config::WARMUP; i++) { 
            CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&p), Config::SIZE_LARGE * sizeof(float))); 
            CUARENA_CHECK(cudaFree(p)); 
        }
        // timed runs with sync single
        std::vector<double> t_sync(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&p), Config::SIZE_LARGE * sizeof(float)));
            CUARENA_CHECK(cudaFree(p));
            t_sync[i] = elapsed(start);
        }

        std::vector<double> t_async;
        if (is_device) {
            // warmup with async single
            for (size_t i = 0; i < Config::WARMUP; i++) { 
                CUARENA_CHECK(cudaMallocAsync(&p, Config::SIZE_LARGE * sizeof(float), stream)); 
                CUARENA_CHECK(cudaFreeAsync(p, stream)); 
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
            // timed runs with async single
            t_async.resize(Config::ITERS);
            for (size_t i = 0; i < Config::ITERS; i++) {
                auto start = Clock::now();
                CUARENA_CHECK(cudaMallocAsync(&p, Config::SIZE_LARGE * sizeof(float), stream));
                CUARENA_CHECK(cudaFreeAsync(p, stream));
                t_async[i] = elapsed(start);
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
        }

        // warmup with arena single
        for (size_t i = 0; i < Config::WARMUP; i++) { 
            auto q = alloc.allocate<float>(Config::SIZE_LARGE); 
            alloc.deallocate(q);
        }
        // timed runs with arena single
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            auto q  = alloc.allocate<float>(Config::SIZE_LARGE);
            alloc.deallocate(q);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_sync, t_async, t_arena, memType.c_str(), areType.c_str(), is_device);
    }
    alloc.reset_gpu_pool();

    // -------------------------------------------------------
    // Benchmark 2: batch allocate 512 x 1 KB, then free all
    // -------------------------------------------------------
    print_header("Benchmark 2: batch alloc 512 x 1 KB, then free all  (per round)");
    {
        std::vector<float*> ptrs(Config::BATCH_SIZE, nullptr);

        // warmpup with sync batch
        for (size_t i = 0; i < Config::WARMUP; i++) {
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&ptrs[j]), Config::SIZE_SMALL * sizeof(float)));
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                CUARENA_CHECK(cudaFree(ptrs[j]));
        }

        // timed runs with sync batch
        std::vector<double> t_sync(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&ptrs[j]), Config::SIZE_SMALL * sizeof(float)));
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                CUARENA_CHECK(cudaFree(ptrs[j]));
            t_sync[i] = elapsed(start);
        }

        std::vector<double> t_async;
        if (is_device) {
            // warmup with async batch
            for (size_t i = 0; i < Config::WARMUP; i++) {
                for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                    CUARENA_CHECK(cudaMallocAsync(&ptrs[j], Config::SIZE_SMALL * sizeof(float), stream));
                for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                    CUARENA_CHECK(cudaFreeAsync(ptrs[j], stream));
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));

            // timed runs with async batch
            t_async.resize(Config::ITERS);
            for (size_t i = 0; i < Config::ITERS; i++) {
                auto start = Clock::now();
                for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                    CUARENA_CHECK(cudaMallocAsync(&ptrs[j], Config::SIZE_SMALL * sizeof(float), stream));
                for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                    CUARENA_CHECK(cudaFreeAsync(ptrs[j], stream));
                t_async[i] = elapsed(start);
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
        }

        // warmup with arena batch
        for (size_t i = 0; i < Config::WARMUP; i++) {
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                ptrs[j] = alloc.allocate<float>(Config::SIZE_SMALL);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                alloc.deallocate(ptrs[j]);
        }

        // timed runs with arena batch
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                ptrs[j] = alloc.allocate<float>(Config::SIZE_SMALL);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++) 
                alloc.deallocate(ptrs[j]);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_sync, t_async, t_arena, memType.c_str(), areType.c_str(), is_device);
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 3: interleaved ping-pong (4 live buffers cycling, 1 MB each)
    // -----------------------------------------------------------------------
    print_header("Benchmark 3: interleaved ping-pong  (4 live x 1 MB, cycling)");
    {
        constexpr size_t LIVE = 4;
        std::vector<float*> live_cuda(LIVE, nullptr);
        std::vector<float*> live_arena(LIVE, nullptr);

        // warmup with sync ping-pong
        for (size_t j = 0; j < LIVE; j++) 
            CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&live_cuda[j]), Config::SIZE_MEDIUM * sizeof(float)));
        for (size_t i = 0; i < Config::WARMUP; i++) {
            size_t slot = i % LIVE;
            CUARENA_CHECK(cudaFree(live_cuda[slot]));
            CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&live_cuda[slot]), Config::SIZE_MEDIUM * sizeof(float)));
        }
        // timed runs with sync ping-pong
        std::vector<double> t_sync(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            size_t slot = i % LIVE;
            auto start = Clock::now();
            CUARENA_CHECK(cudaFree(live_cuda[slot]));
            CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&live_cuda[slot]), Config::SIZE_MEDIUM * sizeof(float)));
            t_sync[i] = elapsed(start);
        }
        for (size_t j = 0; j < LIVE; j++) cudaFree(live_cuda[j]);

        std::vector<double> t_async;
        if (is_device) {
            // warmup with async ping-pong
            for (size_t j = 0; j < LIVE; j++) 
                CUARENA_CHECK(cudaMallocAsync(&live_cuda[j], Config::SIZE_MEDIUM * sizeof(float), stream));
            for (size_t i = 0; i < Config::WARMUP; i++) {
                size_t slot = i % LIVE;
                CUARENA_CHECK(cudaFreeAsync(live_cuda[slot], stream));
                CUARENA_CHECK(cudaMallocAsync(&live_cuda[slot], Config::SIZE_MEDIUM * sizeof(float), stream));
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
            // timed runs with async ping-pong
            t_async.resize(Config::ITERS);
            for (size_t i = 0; i < Config::ITERS; i++) {
                size_t slot = i % LIVE;
                auto start = Clock::now();
                CUARENA_CHECK(cudaFreeAsync(live_cuda[slot], stream));
                CUARENA_CHECK(cudaMallocAsync(&live_cuda[slot], Config::SIZE_MEDIUM * sizeof(float), stream));
                t_async[i] = elapsed(start);
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
            for (size_t j = 0; j < LIVE; j++) 
                CUARENA_CHECK(cudaFreeAsync(live_cuda[j], stream));
            CUARENA_CHECK(cudaStreamSynchronize(stream));
        }

        // warmup with arena ping-pong
        for (size_t j = 0; j < LIVE; j++) 
            live_arena[j] = alloc.allocate<float>(Config::SIZE_MEDIUM);
        for (size_t i = 0; i < Config::WARMUP; i++) {
            size_t slot = i % LIVE;
            alloc.deallocate(live_arena[slot]);
            live_arena[slot] = alloc.allocate<float>(Config::SIZE_MEDIUM);
        }
        // timed runs with arena ping-pong
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            size_t slot = i % LIVE;
            auto start = Clock::now();
            alloc.deallocate(live_arena[slot]);
            live_arena[slot] = alloc.allocate<float>(Config::SIZE_MEDIUM);
            t_arena[i] = elapsed(start);
        }
        for (size_t j = 0; j < LIVE; j++) alloc.deallocate(live_arena[j]);

        print_stats(t_sync, t_async, t_arena, memType.c_str(), areType.c_str(), is_device);
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 4: resize chain  (1 MB to 8 MB in 8 doublings, per iteration)
    // -----------------------------------------------------------------------
    print_header("Benchmark 4: resize chain  (1 MB to 8 MB in 8 doublings)");
    {
        constexpr size_t STEPS = 8;
        // warmup with sync resize chain
        for (size_t i = 0; i < Config::WARMUP; i++) {
            float* p = nullptr; 
            size_t sz = Config::SIZE_MEDIUM;
            for (size_t s = 0; s < STEPS; s++) { 
                CUARENA_CHECK(cudaFree(p)); 
                CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&p), sz * sizeof(float))); sz *= 2; 
            }
            CUARENA_CHECK(cudaFree(p));
        }
        // timed runs with sync resize chain
        std::vector<double> t_sync(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            float* p = nullptr; 
            size_t sz = Config::SIZE_MEDIUM;
            auto start = Clock::now();
            for (size_t s = 0; s < STEPS; s++) { 
                CUARENA_CHECK(cudaFree(p)); 
                CUARENA_CHECK(allocfunc(reinterpret_cast<void**>(&p), sz * sizeof(float))); sz *= 2;
            }
            CUARENA_CHECK(cudaFree(p));
            t_sync[i] = elapsed(start);
        }

        std::vector<double> t_async;
        if (is_device) {
            // warmup with async resize chain
            for (size_t i = 0; i < Config::WARMUP; i++) {
                float* p = nullptr; 
                size_t sz = Config::SIZE_MEDIUM;
                for (size_t s = 0; s < STEPS; s++) { 
                    CUARENA_CHECK(cudaFreeAsync(p, stream)); 
                    CUARENA_CHECK(cudaMallocAsync(&p, sz * sizeof(float), stream)); sz *= 2; 
                }
                CUARENA_CHECK(cudaFreeAsync(p, stream));
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
            // timed runs with async resize chain
            t_async.resize(Config::ITERS);
            for (size_t i = 0; i < Config::ITERS; i++) {
                float* p = nullptr; 
                size_t sz = Config::SIZE_MEDIUM;
                auto start = Clock::now();
                for (size_t s = 0; s < STEPS; s++) { 
                    CUARENA_CHECK(cudaFreeAsync(p, stream)); 
                    CUARENA_CHECK(cudaMallocAsync(&p, sz * sizeof(float), stream)); sz *= 2; 
                }
                CUARENA_CHECK(cudaFreeAsync(p, stream));
                t_async[i] = elapsed(start);
            }
            CUARENA_CHECK(cudaStreamSynchronize(stream));
        }

        // warmup with arena resize chain
        for (size_t i = 0; i < Config::WARMUP; i++) {
            float* p = nullptr; 
            size_t sz = Config::SIZE_MEDIUM;
            for (size_t s = 0; s < STEPS; s++) { alloc.resize<float>(p, sz); sz *= 2; }
            alloc.deallocate(p);
        }
        // timed runs with arena resize chain
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            float* p = nullptr; 
            size_t sz = Config::SIZE_MEDIUM;
            auto start = Clock::now();
            for (size_t s = 0; s < STEPS; s++) { alloc.resize<float>(p, sz); sz *= 2; }
            alloc.deallocate(p);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_sync, t_async, t_arena, memType.c_str(), areType.c_str(), is_device);
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 5: fragmentation + compact cycle
    //   8 x 32 MB allocs, free alternating => 4 x 32 MB holes
    //   Measure: compact latency, then 64 MB alloc that only fits after compact
    // -----------------------------------------------------------------------
    if (is_device) {
        print_header("Benchmark 5: fragmentation + compact cycle  (8 x 32 MB, free alternating)");
        {
            constexpr size_t NBLOCKS  = 8;
            constexpr size_t BLK_ELEM = 32 * cuArena::MB / sizeof(float);

            std::vector<double> t_compact(Config::ITERS);
            std::vector<double> t_alloc_after(Config::ITERS);

            for (size_t i = 0; i < Config::ITERS + Config::WARMUP; i++) {
                // Build fragmented region
                float* ptrs[NBLOCKS];
                for (size_t b = 0; b < NBLOCKS; b++)
                    ptrs[b] = alloc.allocate<float>(BLK_ELEM);
                for (size_t b = 0; b < NBLOCKS; b += 2)
                    alloc.deallocate(ptrs[b]);

                // Time compact
                auto start0 = Clock::now();
                alloc.compact_gpu_dynamic([](cuArena::addr_t, cuArena::addr_t, size_t) {}, stream);
                CUARENA_CHECK(cudaStreamSynchronize(stream));
                double ct = elapsed(start0);

                // Time the large alloc that only succeeds after compact
                auto start1 = Clock::now();
                float* big = alloc.allocate<float>(64 * cuArena::MB / sizeof(float));
                double at = elapsed(start1);

                if (i >= Config::WARMUP) {
                    t_compact[i - Config::WARMUP]     = ct;
                    t_alloc_after[i - Config::WARMUP] = at;
                }

                alloc.reset_gpu_pool();
            }

            auto sc = compute_stats(t_compact);
            auto sa = compute_stats(t_alloc_after);
            print_row("compaction (256 MB active)", sc);
            print_row("alloc 64 MB after compact ",          sa);
            print_separator();
            std::cout << std::format("  {}{:<56}  {:.2f} us{}\n",
                CBENCH, "compact + alloc total (mean)", sc.mean + sa.mean, CUAR_CNORMAL);
            print_separator('=');
            std::cout << '\n';
        }
    }

    // -----------------------------------------------------------------------
    // Benchmark 6: stable vs dynamic allocation latency
    // -----------------------------------------------------------------------
    if (is_device) {
        print_header("Benchmark 6: stable vs dynamic alloc latency  (varied sizes)");
        {
            // Re-create pool with a stable region
            alloc.destroy_gpu_pool();
            alloc.create_gpu_pool(4 * cuArena::GB, cuArena::GPUMemoryType::Device, stream,
                                  512 * cuArena::MB);
            CUARENA_CHECK(cudaStreamSynchronize(stream));

            const size_t sizes[] = { Config::SIZE_SMALL, Config::SIZE_MEDIUM, Config::SIZE_LARGE };
            const char*  labels[] = { "256 B", "256 KB", "2 MB" };

            for (size_t si = 0; si < 3; si++) {
                const size_t ELEM = sizes[si] / sizeof(float);
                std::vector<double> t_stable(Config::ITERS);
                std::vector<double> t_dynamic(Config::ITERS);

                // Warmup stable
                for (size_t i = 0; i < Config::WARMUP; i++) {
                    auto p = alloc.allocate<float>(ELEM, cuArena::Region::Stable);
                    alloc.deallocate(p);
                }
                // Timed stable
                for (size_t i = 0; i < Config::ITERS; i++) {
                    auto start = Clock::now();
                    auto p = alloc.allocate<float>(ELEM, cuArena::Region::Stable);
                    alloc.deallocate(p);
                    t_stable[i] = elapsed(start);
                }

                // Warmup dynamic
                for (size_t i = 0; i < Config::WARMUP; i++) {
                    auto p = alloc.allocate<float>(ELEM, cuArena::Region::Dynamic);
                    alloc.deallocate(p);
                }
                // Timed dynamic
                for (size_t i = 0; i < Config::ITERS; i++) {
                    auto start = Clock::now();
                    auto p = alloc.allocate<float>(ELEM, cuArena::Region::Dynamic);
                    alloc.deallocate(p);
                    t_dynamic[i] = elapsed(start);
                }

                auto ss = compute_stats(t_stable);
                auto sd = compute_stats(t_dynamic);
                std::string ls = std::string("stable  ") + labels[si];
                std::string ld = std::string("dynamic ") + labels[si];
                print_row(ls.c_str(),  ss);
                print_row(ld.c_str(),  sd);
                print_separator();
                const bool dyn_faster = sd.mean < ss.mean;
                const double ratio = dyn_faster ? ss.mean / sd.mean : sd.mean / ss.mean;
                std::cout << std::format("dynamic vs stable [{:<6}]         {}  {:.2f}x {}" CUAR_CNORMAL "\n",
                    labels[si], dyn_faster ? CFASTER : CSLOWER, ratio, dyn_faster ? "faster" : "slower");
                print_separator('=');
                std::cout << '\n';
            }
        }
    }
}
