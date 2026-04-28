#pragma once

#include <mutex>
#include <cstdio>
#include "color.hpp"

namespace cuarena {

    class Logger {

        Logger() = default;

        int        _level = 0;
        std::mutex _mutex;

        static Logger& get() {
            static Logger instance;
            return instance;
        }

    public:

        static void set_level(const int& level) noexcept { get()._level = level; }
        static int  level()                     noexcept { return get()._level; }

        template<typename... Args>
        static void error(const char* fmt, Args&&... args) {
            std::lock_guard<std::mutex> lock(get()._mutex);
            std::fflush(stdout);
            std::fprintf(stderr, "%sERROR: ", CERROR);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stderr, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stderr);
            std::fprintf(stderr, "%s\n", CNORMAL);
            std::fflush(stderr);
        }

        template<typename... Args>
        static void warn(const char* fmt, Args&&... args) {
            std::lock_guard<std::mutex> lock(get()._mutex);
            std::fflush(stdout);
            std::fprintf(stderr, "%sWARNING: ", CWARNING);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stderr, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stderr);
            std::fprintf(stderr, "%s\n", CNORMAL);
            std::fflush(stderr);
        }

        template<typename... Args>
        static void info(const char* fmt, Args&&... args) {
            if (get()._level >= 1) {
                std::lock_guard<std::mutex> lock(get()._mutex);
                std::fprintf(stdout, "%s", CINFO);
                if constexpr (sizeof...(Args) > 0)
                    std::fprintf(stdout, fmt, std::forward<Args>(args)...);
                else
                    std::fputs(fmt, stdout);
                std::fprintf(stdout, "%s\n", CNORMAL);
                std::fflush(stdout);
            }
        }

        template<typename... Args>
        static void debug(const char* fmt, Args&&... args) {
            if (get()._level >= 2) {
                std::lock_guard<std::mutex> lock(get()._mutex);
                std::fprintf(stdout, "%sDEBUG: ", CDEBUG);
                if constexpr (sizeof...(Args) > 0)
                    std::fprintf(stdout, fmt, std::forward<Args>(args)...);
                else
                    std::fputs(fmt, stdout);
                std::fprintf(stdout, "%s\n", CNORMAL);
                std::fflush(stdout);
            }
        }

    };

}
