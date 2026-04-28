#pragma once

#include <unordered_map>
#include <map>
#include <set>
#include <mutex>
#include <cassert>
#include <stdexcept>

#include "logger.hpp"
#include "sysinfo.hpp"
#include "constants.hpp"
#include "definitions.cuh"

namespace cuarena {

    using addr_t = void*;

    struct Pool {
        addr_t mem  = nullptr;
        size_t size = 0;
        size_t cap  = 0;
        size_t off  = 0;
    };

    struct gpu_memory_error : std::runtime_error {
        explicit gpu_memory_error(const char* msg) : std::runtime_error(msg) {}
    };

    struct cpu_memory_error : std::runtime_error {
        explicit cpu_memory_error(const char* msg) : std::runtime_error(msg) {}
    };

    class DeviceArena {

        static constexpr size_t ALIGNMENT   = 128;
        static constexpr size_t MIN_SPLIT   = ALIGNMENT;
        static constexpr size_t GPU_PENALTY = 256 * MB;
        static constexpr size_t CPU_PENALTY = 512 * MB;

        Pool   _gpool, _cpool;
        size_t _gtot = 0, _gfree_mem = 0, _glimit = 0;
        size_t _ctot = 0, _cfree_mem = 0, _climit = 0;
        size_t _gpu_allocated = 0, _cpu_allocated = 0;

        mutable std::mutex _mutex;

        std::map<addr_t, size_t>           _gpu_free_by_addr;
        std::map<size_t, std::set<addr_t>> _gpu_free_by_size;
        std::unordered_map<addr_t, size_t> _gpu_alloc_list;

        std::map<addr_t, size_t>           _cpu_free_by_addr;
        std::map<size_t, std::set<addr_t>> _cpu_free_by_size;
        std::unordered_map<addr_t, size_t> _cpu_alloc_list;

        size_t align_up         (size_t n) const noexcept { return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1); }
        bool   is_aligned       (size_t n) const noexcept { return (n & (ALIGNMENT - 1)) == 0; }

        void   _insert_gpu_free (const addr_t& ptr, const size_t& size);
        void   _insert_cpu_free (const addr_t& ptr, const size_t& size);
        addr_t _pop_gpu_free    (const size_t& size, size_t& out_size);
        addr_t _pop_cpu_free    (const size_t& size, size_t& out_size);
        addr_t _gpu_alloc       (const size_t& aligned_bytes);
        addr_t _cpu_alloc       (const size_t& aligned_bytes);
        void   _gpu_free_block  (const addr_t& ptr);
        void   _cpu_free_block  (const addr_t& ptr);

    public:

        DeviceArena           () = default;
        DeviceArena           (const size_t& cpu_limit, const size_t& gpu_limit, const cudaStream_t& stream = 0);

        DeviceArena           (const DeviceArena&) = delete;
        DeviceArena& operator=(const DeviceArena&) = delete;
        DeviceArena           (DeviceArena&&)      = delete;
        DeviceArena& operator=(DeviceArena&&)      = delete;

        ~DeviceArena();

        bool create_gpu_pool    (const size_t& limit = 0, cudaStream_t stream = 0);
        bool create_cpu_pool    (const size_t& limit = 0);
        bool destroy_gpu_pool   ();
        bool destroy_cpu_pool   ();
        bool resize_gpu_pool    (const size_t& new_size, cudaStream_t stream = 0);
        bool resize_cpu_pool    (const size_t& new_size);
        void reset_gpu_pool     ();
        void reset_cpu_pool     ();

        template<class T>
        T* allocate(const size_t& count) {
            if (!count) return nullptr;
            const size_t aligned = align_up(count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            return static_cast<T*>(_gpu_alloc(aligned));
        }

        template<class T>
        void deallocate(T* ptr) {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_block(static_cast<addr_t>(ptr));
        }

        template<class T>
        void resize(T*& ptr, const size_t& new_count) {
            if (!new_count) {
                deallocate(ptr);
                ptr = nullptr;
                return;
            }
            const size_t aligned = align_up(new_count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (ptr) {
                auto it = _gpu_alloc_list.find(static_cast<addr_t>(ptr));
                if (it == _gpu_alloc_list.end()) {
                    Logger::error("resize: pointer %p not in GPU alloc list", static_cast<addr_t>(ptr));
                    throw gpu_memory_error("invalid pointer");
                }
                if (it->second >= aligned) return;
                _gpu_free_block(static_cast<addr_t>(ptr));
                ptr = nullptr;
            }
            ptr = static_cast<T*>(_gpu_alloc(aligned));
        }

        template<class T>
        T* allocate_pinned(const size_t& count) {
            if (!count) return nullptr;
            const size_t aligned = align_up(count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            return static_cast<T*>(_cpu_alloc(aligned));
        }

        template<class T>
        void deallocate_pinned(T* ptr) {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_block(static_cast<addr_t>(ptr));
        }

        template<class T>
        void resize_pinned(T*& ptr, const size_t& new_count) {
            if (!new_count) {
                deallocate_pinned(ptr);
                ptr = nullptr;
                return;
            }
            const size_t aligned = align_up(new_count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (ptr) {
                auto it = _cpu_alloc_list.find(static_cast<addr_t>(ptr));
                if (it == _cpu_alloc_list.end()) {
                    Logger::error("resize_pinned: pointer %p not in CPU alloc list", static_cast<addr_t>(ptr));
                    throw cpu_memory_error("invalid pointer");
                }
                if (it->second >= aligned) return;
                _cpu_free_block(static_cast<addr_t>(ptr));
                ptr = nullptr;
            }
            ptr = static_cast<T*>(_cpu_alloc(aligned));
        }

        size_t gpu_capacity     () const noexcept { return _gpool.size; }
        size_t gpu_used         () const noexcept { std::lock_guard<std::mutex> lock(_mutex); return _gpu_allocated; }
        size_t gpu_available    () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t freed = 0;
            for (auto& [sz, addrs] : _gpu_free_by_size) 
                freed += sz * addrs.size();
            return _gpool.cap + freed;
        }

        size_t cpu_capacity     () const noexcept { return _cpool.size; }
        size_t cpu_used         () const noexcept { std::lock_guard<std::mutex> lock(_mutex); return _cpu_allocated; }
        size_t cpu_available    () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t freed = 0;
            for (auto& [sz, addrs] : _cpu_free_by_size) 
                freed += sz * addrs.size();
            return _cpool.cap + freed;
        }

    };

}
