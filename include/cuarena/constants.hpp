#pragma once

#include <cstddef>

namespace cuArena {

constexpr size_t KB = 0x00000400ULL;
constexpr size_t MB = 0x00100000ULL;
constexpr size_t GB = 0x40000000ULL;

#define CUARENA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define CUARENA_MAX(x, y) ((x) > (y) ? (x) : (y))

constexpr double ratio(double x, double y) noexcept { return y ? x / y : 0.0; }
constexpr size_t ratio(size_t x, size_t y) noexcept { return y ? x / y : 0ULL; }

}
