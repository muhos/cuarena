#pragma once

#include <cstddef>
#include <cstdint>

namespace cuArena {

    using addr_t  = void*;
    using byte_t  = unsigned char;
    using uint32  = unsigned int;
    using int64   = signed long long int;
    using uint64  = unsigned long long int;

    enum class GPUMemoryType { Device, Managed };
    enum class CPUMemoryType { Pinned, Pageable };
    enum class Region        { Stable, Dynamic  };

}
