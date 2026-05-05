#include <cuarena/sysinfo.hpp>

namespace cuArena {

    size_t sys_mem_used() {
        size_t used = 0;
    #if defined(__linux__) || defined(__CYGWIN__)
        long rss = 0;
        FILE* fp = fopen("/proc/self/statm", "r");
        if (fp) {
            if (fscanf(fp, "%*s%ld", &rss) == 1)
                used = static_cast<size_t>(rss) * static_cast<size_t>(sysconf(_SC_PAGESIZE));
            fclose(fp);
        }
    #elif defined(_WIN32)
        PROCESS_MEMORY_COUNTERS_EX info;
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&info), sizeof(info)))
            used = info.WorkingSetSize;
    #endif
        return used;
    }

    void sys_mem_info(size_t& free_bytes, size_t& total_bytes) {
        free_bytes = total_bytes = 0;
    #if defined(__linux__) || defined(__CYGWIN__)
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            free_bytes  = static_cast<size_t>(si.freeram)  * si.mem_unit;
            total_bytes = static_cast<size_t>(si.totalram) * si.mem_unit;
        }
    #elif defined(_WIN32)
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            free_bytes  = static_cast<size_t>(ms.ullAvailPhys);
            total_bytes = static_cast<size_t>(ms.ullTotalPhys);
        }
    #endif
    }

}
