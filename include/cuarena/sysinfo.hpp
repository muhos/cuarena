#pragma once

#include <cstddef>
#include <cstdio>
#include <iostream>

#if defined(__linux__) || defined(__CYGWIN__)
#  include <sys/sysinfo.h>
#  include <unistd.h>
#elif defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <psapi.h>
#endif

namespace cuArena {

    size_t sys_mem_used();
    void   sys_mem_info(size_t& free_bytes, size_t& total_bytes);

}
