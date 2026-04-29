#include <cuarena/allocator.cuh>

namespace cuarena {

    DeviceArena::DeviceArena(const size_t& cpu_limit, 
                             const size_t& gpu_limit, 
                             const GPUMemoryType& gtype,
                             const CPUMemoryType& ctype,
                             const cudaStream_t& stream) {
        if (!create_cpu_pool(cpu_limit, ctype))
            throw cpu_memory_error("failed to create CPU pinned pool");
        if (!create_gpu_pool(gpu_limit, gtype, stream))
            throw gpu_memory_error("failed to create GPU pool");
    }

    DeviceArena::~DeviceArena() {
        destroy_gpu_pool();
        destroy_cpu_pool();
    }

    bool DeviceArena::create_gpu_pool(const size_t& limit, const GPUMemoryType& type, const cudaStream_t& stream) {
        if (_gpool.mem) {
            Logger::info("GPU pool already exists (%zu MB)", _gpool.size / MB);
            return true;
        }
        cudaMemGetInfo(&_gfree_mem, &_gtot);
        if (_gfree_mem > GPU_PENALTY) _gfree_mem -= GPU_PENALTY;
        _glimit = limit ? align_up(limit) : 0;
        if (_glimit) {
            if (_gfree_mem && _glimit > _gfree_mem) {
                Logger::warn("GPU pool: requested %zu MB, only %zu MB free - capping", _glimit / MB, _gfree_mem / MB);
                _gpool.size = _gfree_mem;
            }
            else {
                _gpool.size = _glimit;
            }
        }
        else {
            if (!_gfree_mem) {
                Logger::error("GPU pool: no free GPU memory available");
                return false;
            }
            _gpool.size = _gfree_mem;
        }
        _gpool.cap = _gpool.size;
        _gpool.off = 0;
        _gtype   = type;
        _gstream = stream;
        if (type == GPUMemoryType::Managed) {
            if (stream) Logger::warn("GPU pool: stream argument ignored for Managed memory");
            if (cudaMallocManaged(reinterpret_cast<void**>(&_gpool.mem), _gpool.size) != cudaSuccess) {
                _gpool = Pool{};
                return false;
            }
            CUARENA_CHECK(cudaMemset(_gpool.mem, 0, _gpool.size));
        }
        else {
            if (CUARENA_MALLOC(reinterpret_cast<void**>(&_gpool.mem), _gpool.size, stream) != cudaSuccess) {
                _gpool = Pool{};
                return false;
            }
            CUARENA_CHECK(cudaMemsetAsync(_gpool.mem, 0, _gpool.size, stream));
        }
        Logger::info("GPU %s pool created: %zu MB",
            _gtype == GPUMemoryType::Device ? "device" : "managed",
            _gpool.size / MB);
        return true;
    }

    bool DeviceArena::create_cpu_pool(const size_t& limit, const CPUMemoryType& type) {
        if (_cpool.mem) {
            Logger::info("CPU pool already exists (%zu MB)", _cpool.size / MB);
            return true;
        }
        sys_mem_info(_cfree_mem, _ctot);
        if (_cfree_mem > CPU_PENALTY) _cfree_mem -= CPU_PENALTY;
        _climit = limit ? align_up(limit) : 0;
        if (_climit) {
            if (_cfree_mem && _climit > _cfree_mem) {
                Logger::warn("CPU pool: requested %zu MB, only %zu MB free — capping", _climit / MB, _cfree_mem / MB);
                _cpool.size = _cfree_mem;
            }
            else {
                _cpool.size = _climit;
            }
        }
        else {
            if (!_cfree_mem) _cfree_mem = 4ULL * GB;
            _cpool.size = _cfree_mem;
        }
        _cpool.cap = _cpool.size;
        _cpool.off = 0;
        _ctype = type;
        if (type == CPUMemoryType::Pageable) {
            _cpool.mem = std::malloc(_cpool.size);
            if (!_cpool.mem) {
                _cpool = Pool{};
                return false;
            }
            std::memset(_cpool.mem, 0, _cpool.size);
        }
        else {
            if (cudaMallocHost(&_cpool.mem, _cpool.size) != cudaSuccess) {
                _cpool = Pool{};
                return false;
            }
        }
        Logger::info("CPU %s pool created: %zu MB", 
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable",
            _cpool.size / MB);
        return true;
    }

    bool DeviceArena::destroy_gpu_pool() {
        if (!_gpool.mem) return true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            _gpu_free_by_size.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
        }
        CUARENA_CHECK(CUARENA_FREE(_gpool.mem, _gstream));
        CUARENA_CHECK(cudaStreamSynchronize(_gstream));
        _gpool = Pool{};
        Logger::info("GPU %s pool destroyed",
            _gtype == GPUMemoryType::Device ? "device" : "managed");
        return true;
    }

    bool DeviceArena::destroy_cpu_pool() {
        if (!_cpool.mem) return true;
        { // Limit lock's scope
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
        }
        if (_ctype == CPUMemoryType::Pageable) {
            std::free(_cpool.mem);
        }
        else {
            if (cudaFreeHost(_cpool.mem) != cudaSuccess) return false;
        }
        _cpool = Pool{};
        Logger::info("CPU %s pool destroyed",
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable");
        return true;
    }

    bool DeviceArena::resize_gpu_pool(const size_t& new_size, cudaStream_t stream) {
        if (!_gpool.mem) {
            Logger::warn("GPU pool: resize called but pool not yet created");
            return false;
        }
        cudaMemGetInfo(&_gfree_mem, &_gtot);
        if (new_size > _gfree_mem) {
            Logger::warn("Resizing GPU pool: %zu MB requested but only %zu MB free", new_size / MB, _gfree_mem / MB);
        }
        { // Limit lock's scope
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            _gpu_free_by_size.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            CUARENA_FREE(_gpool.mem, _gstream);
            cudaStreamSynchronize(_gstream);
            _gpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _gpool.size = _gpool.cap = aligned;
        _glimit  = aligned;
        _gstream = stream;
        if (_gtype == GPUMemoryType::Managed) {
            if (stream) Logger::warn("Resizing GPU pool: stream argument ignored for Managed memory");
            if (cudaMallocManaged(reinterpret_cast<void**>(&_gpool.mem), _gpool.size) != cudaSuccess) {
                _gpool = Pool{};
                return false;
            }
            CUARENA_CHECK(cudaMemset(_gpool.mem, 0, _gpool.size));
        }
        else {
            if (CUARENA_MALLOC(reinterpret_cast<void**>(&_gpool.mem), _gpool.size, stream) != cudaSuccess) {
                _gpool = Pool{};
                return false;
            }
            CUARENA_CHECK(cudaMemsetAsync(_gpool.mem, 0, _gpool.size, stream));
        }
        Logger::info("GPU %s pool resized to %zu MB", 
            _gtype == GPUMemoryType::Device ? "device" : "managed",
            _gpool.size / MB);
        return true;
    }

    bool DeviceArena::resize_cpu_pool(const size_t& new_size) {
        if (!_cpool.mem) {
            Logger::warn("Resizing CPU pool: pool not yet created");
            return false;
        }
        sys_mem_info(_cfree_mem, _ctot);
        if (new_size > _cfree_mem) {
            Logger::warn("Resizing CPU pool: %zu MB requested but only %zu MB free", new_size / MB, _cfree_mem / MB);
        }
        { // Limit lock's scope
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
            cudaFreeHost(_cpool.mem);
            _cpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _cpool.size = _cpool.cap = aligned;
        _climit = aligned;
        if (_ctype == CPUMemoryType::Pageable) {
            _cpool.mem = std::malloc(_cpool.size);
            if (!_cpool.mem) {
                _cpool = Pool{};
                return false;
            }
            std::memset(_cpool.mem, 0, _cpool.size);
        }
        else {
            if (cudaMallocHost(&_cpool.mem, _cpool.size) != cudaSuccess) {
                _cpool = Pool{};
                return false;
            }
        }
        Logger::info("CPU %s pool resized to %zu MB", 
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable",
            _cpool.size / MB);
        return true;
    }

    void DeviceArena::reset_gpu_pool() {
        std::lock_guard<std::mutex> lock(_mutex);
        _gpu_free_by_addr.clear();
        _gpu_free_by_size.clear();
        _gpu_alloc_list.clear();
        _gpu_allocated = 0;
        _gpool.off = 0;
        _gpool.cap = _gpool.size;
    }

    void DeviceArena::reset_cpu_pool() {
        std::lock_guard<std::mutex> lock(_mutex);
        _cpu_free_by_addr.clear();
        _cpu_free_by_size.clear();
        _cpu_alloc_list.clear();
        _cpu_allocated = 0;
        _cpool.off = 0;
        _cpool.cap = _cpool.size;
    }

    void DeviceArena::_insert_gpu_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _gpu_free_by_addr.emplace(ptr, size);
        assert(ok);

        // Try to coalesce with next block
        auto next = std::next(it);
        if (next != _gpu_free_by_addr.end()) {
            if (static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
                auto& s = _gpu_free_by_size[next->second];
                s.erase(next->first);
                if (s.empty()) _gpu_free_by_size.erase(next->second);
                it->second += next->second;
                _gpu_free_by_addr.erase(next);
            }
        }

        if (it != _gpu_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                auto& s = _gpu_free_by_size[prev->second];
                s.erase(prev->first);
                if (s.empty()) _gpu_free_by_size.erase(prev->second);
                prev->second += it->second;
                _gpu_free_by_addr.erase(it);
                it = prev;
            }
        }

        _gpu_free_by_size[it->second].insert(it->first);
    }

    void DeviceArena::_insert_cpu_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _cpu_free_by_addr.emplace(ptr, size);
        assert(ok);

        // Try to coalesce with next block
        auto next = std::next(it);
        if (next != _cpu_free_by_addr.end()) {
            if (static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
                auto& s = _cpu_free_by_size[next->second];
                s.erase(next->first);
                if (s.empty()) _cpu_free_by_size.erase(next->second);
                it->second += next->second;
                _cpu_free_by_addr.erase(next);
            }
        }

        if (it != _cpu_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                auto& s = _cpu_free_by_size[prev->second];
                s.erase(prev->first);
                if (s.empty()) _cpu_free_by_size.erase(prev->second);
                prev->second += it->second;
                _cpu_free_by_addr.erase(it);
                it = prev;
            }
        }

        _cpu_free_by_size[it->second].insert(it->first);
    }

    addr_t DeviceArena::_pop_gpu_free(const size_t& size, size_t& out_size) {
        auto it = _gpu_free_by_size.lower_bound(size);
        if (it == _gpu_free_by_size.end()) { out_size = 0; return nullptr; }

        addr_t block = *it->second.begin();
        const size_t block_size = it->first;

        it->second.erase(it->second.begin());
        if (it->second.empty()) _gpu_free_by_size.erase(it);
        _gpu_free_by_addr.erase(block);

        if (block_size >= size + MIN_SPLIT) {
            addr_t rem = static_cast<byte_t*>(block) + size;
            const size_t rem_size = block_size - size;
            _gpu_free_by_addr.emplace(rem, rem_size);
            _gpu_free_by_size[rem_size].insert(rem);
            out_size = size;
        }
        else {
            out_size = block_size;
        }
        return block;
    }

    addr_t DeviceArena::_pop_cpu_free(const size_t& size, size_t& out_size) {
        auto it = _cpu_free_by_size.lower_bound(size);
        if (it == _cpu_free_by_size.end()) { out_size = 0; return nullptr; }

        addr_t block = *it->second.begin();
        const size_t block_size = it->first;

        it->second.erase(it->second.begin());
        if (it->second.empty()) _cpu_free_by_size.erase(it);
        _cpu_free_by_addr.erase(block);

        if (block_size >= size + MIN_SPLIT) {
            addr_t rem = static_cast<byte_t*>(block) + size;
            const size_t rem_size = block_size - size;
            _cpu_free_by_addr.emplace(rem, rem_size);
            _cpu_free_by_size[rem_size].insert(rem);
            out_size = size;
        }
        else {
            out_size = block_size;
        }
        return block;
    }

    addr_t DeviceArena::_gpu_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));
        size_t out_size = 0;
        addr_t ptr = _pop_gpu_free(aligned_bytes, out_size);
        if (ptr) {
            _gpu_alloc_list.emplace(ptr, out_size);
            _gpu_allocated += out_size;
            return ptr;
        }
        if (aligned_bytes > _gpool.cap) {
            Logger::error("GPU pool exhausted: %zu bytes requested / %zu bytes available", aligned_bytes, _gpool.cap);
            throw gpu_memory_error("GPU pool out of memory");
        }
        ptr = static_cast<byte_t*>(_gpool.mem) + _gpool.off;
        _gpool.off += aligned_bytes;
        _gpool.cap -= aligned_bytes;
        _gpu_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_allocated += aligned_bytes;
        return ptr;
    }

    addr_t DeviceArena::_cpu_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));
        size_t out_size = 0;
        addr_t ptr = _pop_cpu_free(aligned_bytes, out_size);
        if (ptr) {
            _cpu_alloc_list.emplace(ptr, out_size);
            _cpu_allocated += out_size;
            return ptr;
        }
        if (aligned_bytes > _cpool.cap) {
            Logger::error("CPU pool exhausted: %zu bytes requested / %zu bytes available", aligned_bytes, _cpool.cap);
            throw cpu_memory_error("CPU pool out of memory");
        }
        ptr = static_cast<byte_t*>(_cpool.mem) + _cpool.off;
        _cpool.off += aligned_bytes;
        _cpool.cap -= aligned_bytes;
        _cpu_alloc_list.emplace(ptr, aligned_bytes);
        _cpu_allocated += aligned_bytes;
        return ptr;
    }

    void DeviceArena::_gpu_free_block(const addr_t& ptr) {
        auto it = _gpu_alloc_list.find(ptr);
        if (it == _gpu_alloc_list.end()) {
            Logger::error("deallocate: pointer %p not in GPU alloc list", ptr);
            throw gpu_memory_error("invalid pointer");
        }
        const size_t size = it->second;
        _gpu_alloc_list.erase(it);
        _gpu_allocated -= size;
        _insert_gpu_free(ptr, size);
    }

    void DeviceArena::_cpu_free_block(const addr_t& ptr) {
        auto it = _cpu_alloc_list.find(ptr);
        if (it == _cpu_alloc_list.end()) {
            Logger::error("deallocate pinned: pointer %p not in CPU alloc list", ptr);
            throw cpu_memory_error("invalid pointer");
        }
        const size_t size = it->second;
        _cpu_alloc_list.erase(it);
        _cpu_allocated -= size;
        _insert_cpu_free(ptr, size);
    }

}
