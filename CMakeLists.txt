cmake_minimum_required(VERSION 3.18)

project(cuarena VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CUARENA_SYNC_ALLOC "Use cudaMalloc instead of cudaMallocAsync" OFF)
option(CUARENA_BUILD_EXAMPLES "Build examples" ON)
option(CUARENA_BUILD_TESTS    "Build tests"    ON)

add_library(cuarena STATIC
    src/allocator.cu
    src/sysinfo.cpp
)

target_include_directories(cuarena
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(cuarena PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)

target_compile_options(cuarena PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --generate-line-info>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>: -G>
)

if(CUARENA_SYNC_ALLOC)
    target_compile_definitions(cuarena PUBLIC CUARENA_SYNC_ALLOC)
endif()

if(CUARENA_BUILD_EXAMPLES)
    add_executable(cuarena_usage examples/usage.cu)
    target_link_libraries(cuarena_usage PRIVATE cuarena)
    set_target_properties(cuarena_usage PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(cuarena_benchmark examples/benchmark.cu)
    target_link_libraries(cuarena_benchmark PRIVATE cuarena)
    set_target_properties(cuarena_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(cuarena_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

if(CUARENA_BUILD_TESTS)
    add_executable(cuarena_test tests/test.cu)
    target_link_libraries(cuarena_test PRIVATE cuarena)
    set_target_properties(cuarena_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(cuarena_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

include(GNUInstallDirs)

install(TARGETS cuarena
    EXPORT cuarenaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/cuarena
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT cuarenaTargets
    FILE      cuarenaTargets.cmake
    NAMESPACE cuarena::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuarena
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cuarenaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cuarenaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuarena
)
