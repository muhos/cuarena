#include <cuarena/allocator.cuh>

namespace cuArena {

    DeviceArena::DeviceArena(const size_t& cpu_limit,
                             const size_t& gpu_limit,
                             const GPUMemoryType& gtype,
                             const CPUMemoryType& ctype,
                             const cudaStream_t& stream) {
        if (!create_cpu_pool(cpu_limit, ctype)) {
            std::string type = (ctype == CPUMemoryType::Pinned) ? "pinned" : "pageable";
            std::string msg = "failed to create CPU " + type + " pool";
            throw cpu_memory_error(msg.c_str());
        }
        if (!create_gpu_pool(gpu_limit, gtype, stream)) {
            std::string type = (gtype == GPUMemoryType::Device) ? "device" : "managed";
            std::string msg = "failed to create GPU " + type + " pool";
            throw gpu_memory_error(msg.c_str());
        }
    }

    DeviceArena::~DeviceArena() {
        destroy_gpu_pool();
        destroy_cpu_pool();
    }

    bool DeviceArena::create_gpu_pool(const size_t& limit,
                                      const GPUMemoryType& type,
                                      const cudaStream_t& stream,
                                      const size_t& stable_bytes) {
        if (_gpool.mem) {
            Logger::info("GPU pool already exists (%zu MB)", _gpool.size / MB);
            return true;
        }
        cudaMemGetInfo(&_gfree_mem, &_gtot);
        if (_gfree_mem > GPU_PENALTY) _gfree_mem -= GPU_PENALTY;
        _glimit = limit ? align_up(limit) : 0;
        if (_glimit) {
            if (_gfree_mem && _glimit > _gfree_mem) {
                Logger::warn("GPU pool: requested %zu MB, only %zu MB free — capping",
                             _glimit / MB, _gfree_mem / MB);
                _gpool.size = _gfree_mem;
            }
            else {
                _gpool.size = _glimit;
            }
        }
        else {
            if (!_gfree_mem) throw gpu_memory_error("GPU pool: no free GPU memory available");
            _gpool.size = _gfree_mem;
        }

        if (stable_bytes > 0) {
            _stable_size = align_up(stable_bytes);
            if (_stable_size >= _gpool.size) {
                Logger::warn("GPU pool: stable bytes (%zu MB) exceeds pool size (%zu MB) - ignoring stable region",
                             stable_bytes / MB, _gpool.size / MB);
                _stable_size  = 0;
                _stable_bytes = 0;
            }
            else {
                _stable_bytes = _stable_size;
            }
        }
        else {
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        _stable_off = 0;
        _stable_cap = _stable_size;
        _gpool.off = _stable_size; // Dynamic region starts right after stable region
        _gpool.cap = _gpool.size - _stable_size;
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

        Logger::info("GPU %s pool created: %zu MB (stable: %zu MB, dynamic: %zu MB)",
            _gtype == GPUMemoryType::Device ? "device" : "managed",
            _gpool.size / MB, _stable_size / MB, (_gpool.size - _stable_size) / MB);
        return true;
    }

    bool DeviceArena::destroy_gpu_pool() {
        if (!_gpool.mem) return true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            _gpu_stable_free_by_addr.clear();
            _gpu_stable_alloc_list.clear();
            _gpu_stable_allocated = 0;
            _stable_off   = 0;
            _stable_cap   = 0;
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        CUARENA_CHECK(CUARENA_FREE(_gpool.mem, _gstream));
        CUARENA_CHECK(cudaStreamSynchronize(_gstream));
        _gpool = Pool{};
        Logger::info("GPU %s pool destroyed",
            _gtype == GPUMemoryType::Device ? "device" : "managed");
        return true;
    }

    bool DeviceArena::resize_gpu_pool(const size_t& new_size, cudaStream_t stream) {
        if (!_gpool.mem) {
            Logger::warn("GPU pool: resize called but pool not yet created");
            return false;
        }
        cudaMemGetInfo(&_gfree_mem, &_gtot);
        if (new_size > _gfree_mem) {
            Logger::warn("Resizing GPU pool: %zu MB requested but only %zu MB free",
                         new_size / MB, _gfree_mem / MB);
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            _gpu_stable_free_by_addr.clear();
            _gpu_stable_alloc_list.clear();
            _gpu_stable_allocated = 0;
            CUARENA_FREE(_gpool.mem, _gstream);
            cudaStreamSynchronize(_gstream);
            _gpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _gpool.size  = aligned;
        // Restore exact stable size; warn and drop it if the new pool is too small
        if (_stable_bytes > 0 && _stable_bytes >= aligned) {
            Logger::warn("GPU pool resize: stable region (%zu MB) >= new pool size (%zu MB) - dropping stable region",
                         _stable_bytes / MB, aligned / MB);
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        else {
            _stable_size = _stable_bytes;
        }
        _stable_off  = 0;
        _stable_cap  = _stable_size;
        _gpool.off   = _stable_size;
        _gpool.cap   = aligned - _stable_size;
        _glimit      = aligned;
        _gstream     = stream;

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

    void DeviceArena::reset_gpu_pool() {
        std::lock_guard<std::mutex> lock(_mutex);
        _gpu_free_by_addr.clear();
        for (auto& bin : _gpu_bins) bin.clear();
        _gpu_alloc_list.clear();
        _gpu_allocated = 0;
        _gpu_stable_free_by_addr.clear();
        _gpu_stable_alloc_list.clear();
        _gpu_stable_allocated = 0;
        _stable_off  = 0;
        _stable_cap  = _stable_size;
        _gpool.off   = _stable_size; // dynamic region starts right after stable region
        _gpool.cap   = _gpool.size - _stable_size;
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
                Logger::warn("CPU pool: requested %zu MB, only %zu MB free — capping",
                             _climit / MB, _cfree_mem / MB);
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
            if (!_cpool.mem) { _cpool = Pool{}; return false; }
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

    bool DeviceArena::destroy_cpu_pool() {
        if (!_cpool.mem) return true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
        }
        if (_ctype == CPUMemoryType::Pageable)
            std::free(_cpool.mem);
        else
            if (cudaFreeHost(_cpool.mem) != cudaSuccess) return false;
        _cpool = Pool{};
        Logger::info("CPU %s pool destroyed",
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable");
        return true;
    }

    bool DeviceArena::resize_cpu_pool(const size_t& new_size) {
        if (!_cpool.mem) {
            Logger::warn("Resizing CPU pool: pool not yet created");
            return false;
        }
        sys_mem_info(_cfree_mem, _ctot);
        if (new_size > _cfree_mem) {
            Logger::warn("Resizing CPU pool: %zu MB requested but only %zu MB free",
                         new_size / MB, _cfree_mem / MB);
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
            if (_ctype == CPUMemoryType::Pageable)
                std::free(_cpool.mem);
            else
                cudaFreeHost(_cpool.mem);
            _cpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _cpool.size = _cpool.cap = aligned;
        _climit = aligned;
        if (_ctype == CPUMemoryType::Pageable) {
            _cpool.mem = std::malloc(_cpool.size);
            if (!_cpool.mem) { _cpool = Pool{}; return false; }
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

        // Coalesce with next block
        auto next = std::next(it);
        if (next != _gpu_free_by_addr.end()) {
            if (static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
                _gpu_bins[_bin_index(next->second)].erase(next->first);
                it->second += next->second;
                _gpu_free_by_addr.erase(next);
            }
        }

        // Coalesce with previous block
        if (it != _gpu_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                _gpu_bins[_bin_index(prev->second)].erase(prev->first);
                prev->second += it->second;
                _gpu_free_by_addr.erase(it);
                it = prev;
            }
        }

        // Insert merged block into the appropriate bin
        _gpu_bins[_bin_index(it->second)].insert(it->first);
    }

    addr_t DeviceArena::_pop_gpu_free(const size_t& size, size_t& out_size) {
        int k = _bin_index(size);
        
        // search bin k for a block large enough to satisfy the request
        {
            auto& bin = _gpu_bins[k];
            for (auto addr_it = bin.begin(); addr_it != bin.end(); ++addr_it) {
                auto it = _gpu_free_by_addr.find(*addr_it);
                assert(it != _gpu_free_by_addr.end());
                if (it->second >= size) {
                    addr_t block = it->first;
                    const size_t block_size = it->second;
                    bin.erase(addr_it);
                    _gpu_free_by_addr.erase(it);
                    // Split remainder
                    if (block_size >= size + MIN_SPLIT) {
                        addr_t rem = static_cast<byte_t*>(block) + size;
                        const size_t rsz = block_size - size;
                        _gpu_free_by_addr.emplace(rem, rsz);
                        _gpu_bins[_bin_index(rsz)].insert(rem);
                        out_size = size;
                    } 
                    else {
                        out_size = block_size;
                    }
                    return block;
                }
            }
        }

        // look in higher bins for a block to split
        for (int j = k + 1; j < NUM_BINS; ++j) {
            auto& bin = _gpu_bins[j];
            if (bin.empty()) continue;
            auto addr_it = bin.begin();
            auto it = _gpu_free_by_addr.find(*addr_it);
            assert(it != _gpu_free_by_addr.end());
            addr_t block = it->first;
            const size_t block_size = it->second;
            bin.erase(addr_it);
            _gpu_free_by_addr.erase(it);
            if (block_size >= size + MIN_SPLIT) {
                addr_t rem = static_cast<byte_t*>(block) + size;
                const size_t rsz = block_size - size;
                _gpu_free_by_addr.emplace(rem, rsz);
                _gpu_bins[_bin_index(rsz)].insert(rem);
                out_size = size;
            } 
            else {
                out_size = block_size;
            }
            return block;
        }

        out_size = 0;
        return nullptr;
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
            Logger::debug("cuArena: GPU pool out of memory (dynamic region: %zu MB remaining, requested: %zu MB)",
                         _gpool.cap / MB, aligned_bytes / MB);
            throw gpu_memory_error("GPU pool out of memory");
        }
        ptr = static_cast<byte_t*>(_gpool.mem) + _gpool.off;
        _gpool.off += aligned_bytes;
        _gpool.cap -= aligned_bytes;
        _gpu_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_allocated += aligned_bytes;
        return ptr;
    }

    void DeviceArena::_gpu_free_block(const addr_t& ptr) {
        auto it = _gpu_alloc_list.find(ptr);
        if (it == _gpu_alloc_list.end()) {
            Logger::debug("deallocate: pointer %p not in GPU alloc list", ptr);
            throw gpu_memory_error("invalid pointer in freeing GPU block");
        }
        const size_t size = it->second;
        _gpu_alloc_list.erase(it);
        _gpu_allocated -= size;
        _insert_gpu_free(ptr, size);
    }

    addr_t DeviceArena::_stable_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));

        // First-fit from stable free list
        for (auto it = _gpu_stable_free_by_addr.begin();
             it != _gpu_stable_free_by_addr.end(); ++it) {
            if (it->second >= aligned_bytes) {
                addr_t block = it->first;
                size_t block_size   = it->second;
                _gpu_stable_free_by_addr.erase(it);
                // Split remainder back into free list
                if (block_size >= aligned_bytes + MIN_SPLIT) {
                    addr_t rem = static_cast<byte_t*>(block) + aligned_bytes;
                    _gpu_stable_free_by_addr.emplace(rem, block_size - aligned_bytes);
                    block_size = aligned_bytes;
                }
                _gpu_stable_alloc_list.emplace(block, block_size);
                _gpu_stable_allocated += block_size;
                return block;
            }
        }

        // Stable region allocator
        if (aligned_bytes > _stable_cap) {
            Logger::debug("cuArena: stable GPU region out of memory (%zu MB remaining, requested %zu MB)",
                         _stable_cap / MB, aligned_bytes / MB);
            throw gpu_memory_error("cuArena: stable GPU region out of memory");
        }
        addr_t ptr = static_cast<byte_t*>(_gpool.mem) + _stable_off;
        _stable_off += aligned_bytes;
        _stable_cap -= aligned_bytes;
        _gpu_stable_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_stable_allocated += aligned_bytes;
        return ptr;
    }

    void DeviceArena::_stable_free_block(const addr_t& ptr) {
        auto it = _gpu_stable_alloc_list.find(ptr);
        if (it == _gpu_stable_alloc_list.end()) {
            Logger::debug("deallocate stable: pointer %p not in stable alloc list", ptr);
            throw gpu_memory_error("invalid pointer in freeing stable GPU block");
        }
        const size_t size = it->second;
        _gpu_stable_alloc_list.erase(it);
        _gpu_stable_allocated -= size;

        // Insert with coalescing
        auto [fit, ok] = _gpu_stable_free_by_addr.emplace(ptr, size);
        assert(ok);

        auto next = std::next(fit);
        if (next != _gpu_stable_free_by_addr.end()) {
            if (static_cast<byte_t*>(fit->first) + fit->second == static_cast<byte_t*>(next->first)) {
                fit->second += next->second;
                _gpu_stable_free_by_addr.erase(next);
            }
        }
        if (fit != _gpu_stable_free_by_addr.begin()) {
            auto prev = std::prev(fit);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(fit->first)) {
                prev->second += fit->second;
                _gpu_stable_free_by_addr.erase(fit);
            }
        }
    }

    void DeviceArena::compact_gpu_dynamic(RelocateCallback callback, cudaStream_t stream) {
        std::vector<std::pair<addr_t, size_t>> active_blocks; // (old_ptr, size) sorted by addr
        std::vector<addr_t>                    new_ptrs;
        {
            std::lock_guard<std::mutex> lock(_mutex);

            if (_gpu_alloc_list.empty()) {
                // No dynamic allocs: discard all fragmentation holes by resetting
                // the region pointer to the start of the dynamic region.
                _gpu_free_by_addr.clear();
                for (auto& bin : _gpu_bins) bin.clear();
                _gpool.off = _stable_size;
                _gpool.cap = _gpool.size - _stable_size;
                return;
            }

            // Collect and sort active blocks by ascending address
            active_blocks.assign(_gpu_alloc_list.begin(), _gpu_alloc_list.end());
            std::sort(active_blocks.begin(), active_blocks.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            // Compact: pack from dynamic start forward, no gaps
            byte_t* dyn_start = static_cast<byte_t*>(_gpool.mem) + _stable_size;
            size_t  new_off   = 0;
            new_ptrs.reserve(active_blocks.size());
            for (auto& [old_ptr, sz] : active_blocks) {
                new_ptrs.push_back(dyn_start + new_off);
                new_off += sz;
            }

            // Rebuild alloc list and free list to reflect new packed layout
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            for (size_t i = 0; i < active_blocks.size(); ++i) {
                _gpu_alloc_list.emplace(new_ptrs[i], active_blocks[i].second);
                _gpu_allocated += active_blocks[i].second;
            }
            _gpool.off = _stable_size + new_off;
            _gpool.cap = _gpool.size - _gpool.off;
        }

        // Move data on GPU to new compacted locations.
        for (size_t i = 0; i < active_blocks.size(); ++i) {
            if (new_ptrs[i] != active_blocks[i].first) {
                CUARENA_CHECK(cudaMemcpyAsync(new_ptrs[i], 
                    active_blocks[i].first, active_blocks[i].second, cudaMemcpyDeviceToDevice, stream));
            }
        }

        // Invoke callbacks (CPU-side pointer updates).
        if (callback) {
            for (size_t i = 0; i < active_blocks.size(); ++i) {
                callback(active_blocks[i].first, new_ptrs[i], active_blocks[i].second);
            }
        }
    }

    void DeviceArena::_insert_cpu_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _cpu_free_by_addr.emplace(ptr, size);
        assert(ok);

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
            throw cpu_memory_error("CPU pool out of memory");
        }
        ptr = static_cast<byte_t*>(_cpool.mem) + _cpool.off;
        _cpool.off += aligned_bytes;
        _cpool.cap -= aligned_bytes;
        _cpu_alloc_list.emplace(ptr, aligned_bytes);
        _cpu_allocated += aligned_bytes;
        return ptr;
    }

    void DeviceArena::_cpu_free_block(const addr_t& ptr) {
        auto it = _cpu_alloc_list.find(ptr);
        if (it == _cpu_alloc_list.end()) {
            Logger::debug("deallocate pinned: pointer %p not in CPU alloc list", ptr);
            throw cpu_memory_error("invalid pointer in freeing CPU block");
        }
        const size_t size = it->second;
        _cpu_alloc_list.erase(it);
        _cpu_allocated -= size;
        _insert_cpu_free(ptr, size);
    }

}
