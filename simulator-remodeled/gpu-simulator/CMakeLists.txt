cmake_minimum_required(VERSION 3.17)

project(Modern-GPU-Simulator
        VERSION 1.0.0
        DESCRIPTION "Modern GPU Simulator (MICRO 2025 remodel)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(ModernGpuSimEnvironment)

option(ACCELSIM_CONFIG "Accel-Sim build flavor (release or debug)" "release")
if(NOT ACCELSIM_CONFIG STREQUAL "debug")
    set(ACCELSIM_CONFIG "release")
endif()

modern_gpusim_detect_environment(ACCELSIM_CONFIG "${ACCELSIM_CONFIG}")

if(ACCELSIM_CONFIG STREQUAL "debug")
    set(CMAKE_BUILD_TYPE Debug)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG=1)
    add_compile_options(-Wall -O0 -g3 -fPIC)
else()
    add_compile_definitions(DEBUG=0)
    add_compile_options(-Wall -O3 -g3 -fPIC)
endif()

add_compile_options(-fopenmp)
find_package(OpenMP REQUIRED)

# Prefer protobuf CONFIG (protobuf >= 4 / EasyBuild) for correct Abseil deps
find_package(protobuf CONFIG QUIET)
if(protobuf_FOUND)
    set(_protobuf_lib protobuf::libprotobuf)
else()
    find_package(Protobuf REQUIRED)
    if(TARGET protobuf::libprotobuf)
        set(_protobuf_lib protobuf::libprotobuf)
    else()
        set(_protobuf_lib ${Protobuf_LIBRARIES})
    endif()
endif()
set(GPGPUSIM_BUILD_STAMP "${CMAKE_CURRENT_BINARY_DIR}/gpgpu-sim.stamp")

# Resolve compiler paths (CMake may store bare "gcc"/"g++" on some systems).
set(_gpgpu_cc "${CMAKE_C_COMPILER}")
set(_gpgpu_cxx "${CMAKE_CXX_COMPILER}")
if(NOT IS_ABSOLUTE "${_gpgpu_cc}")
    get_filename_component(_gpgpu_cc "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_gpgpu_cc "${_gpgpu_cc}/gcc")
endif()
if(NOT IS_ABSOLUTE "${_gpgpu_cxx}")
    get_filename_component(_gpgpu_cxx "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_gpgpu_cxx "${_gpgpu_cxx}/g++")
endif()
get_filename_component(_gpgpu_cc_bin_dir "${_gpgpu_cc}" DIRECTORY)
get_filename_component(_gpgpu_cxx_bin_dir "${_gpgpu_cxx}" DIRECTORY)

include(ProcessorCount)
ProcessorCount(_gpgpu_build_jobs)
if(NOT _gpgpu_build_jobs)
    set(_gpgpu_build_jobs 4)
endif()

set(_gpgpu_build_script "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/build_gpgpusim.sh")

add_custom_command(
    OUTPUT ${GPGPUSIM_BUILD_STAMP}
    COMMAND /bin/bash "${_gpgpu_build_script}"
        "${GPGPUSIM_ROOT}"
        "${ACCELSIM_CONFIG}"
        "${CUDA_INSTALL_PATH}"
        "${_gpgpu_cc}"
        "${_gpgpu_cxx}"
        "${_gpgpu_build_jobs}"
    COMMAND ${CMAKE_COMMAND} -E touch ${GPGPUSIM_BUILD_STAMP}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building bundled GPGPU-Sim (Makefile)"
    VERBATIM
)
add_custom_target(gpgpu_sim_build DEPENDS ${GPGPUSIM_BUILD_STAMP})

modern_gpusim_write_version_header("${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../util/traces_enhanced
                 ${CMAKE_CURRENT_BINARY_DIR}/traces_enhanced)
add_subdirectory(trace-driven)
add_subdirectory(trace-parser)

set(GPGPUSIM_LIBCUDART "${GPGPUSIM_LIB_DIR}/libcudart.so")

add_executable(accel-sim.out main.cc)
target_include_directories(accel-sim.out
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ISA_Def
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-driven
        ${CMAKE_CURRENT_SOURCE_DIR}/trace-parser
        ${GPGPUSIM_ROOT}/src
        ${GPGPUSIM_ROOT}/libcuda
        ${CUDA_INSTALL_PATH}/include
)
add_dependencies(accel-sim.out gpgpu_sim_build)
target_link_directories(accel-sim.out PRIVATE ${GPGPUSIM_LIB_DIR})
target_link_libraries(accel-sim.out
    PRIVATE
        trace-driven
        trace-parser
        cudart
        traces_enhanced
        OpenMP::OpenMP_CXX
        ${_protobuf_lib}
        GL
        z
        m
        pthread
)
# Resolve symbols from libcudart.so that depend on traces_enhanced (link order matters).
target_link_options(accel-sim.out PRIVATE "-Wl,--no-as-needed")

set(_bin_dir "${CMAKE_CURRENT_SOURCE_DIR}/bin/${ACCELSIM_CONFIG}")
add_custom_command(TARGET accel-sim.out POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_bin_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:accel-sim.out>"
        "${_bin_dir}/accel-sim.out"
    COMMENT "Copying accel-sim.out to ${_bin_dir}/"
)

install(TARGETS accel-sim.out RUNTIME DESTINATION bin/${ACCELSIM_CONFIG})

message(STATUS "Modern GPU Simulator CMake configuration:")
message(STATUS "  ACCELSIM_CONFIG   = ${ACCELSIM_CONFIG}")
message(STATUS "  CUDA_INSTALL_PATH = ${CUDA_INSTALL_PATH}")
message(STATUS "  GPGPUSIM_ROOT     = ${GPGPUSIM_ROOT}")
message(STATUS "  GPGPUSIM_CONFIG   = ${GPGPUSIM_CONFIG}")
message(STATUS "  Output binary     = ${_bin_dir}/accel-sim.out")
