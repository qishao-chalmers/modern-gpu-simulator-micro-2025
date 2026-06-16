# Shared environment detection for Makefile and CMake builds.

function(_modern_gpusim_normalize_cuda_path _in out_var nvcc_var)
    set(_cuda "${_in}")
    if(EXISTS "${_cuda}/bin/nvcc")
        set(_nvcc "${_cuda}/bin/nvcc")
    elseif(EXISTS "/usr/bin/nvcc")
        set(_nvcc "/usr/bin/nvcc")
        # GPGPU-Sim Makefiles expect $CUDA_INSTALL_PATH/bin/nvcc
        set(_cuda_shim "${CMAKE_BINARY_DIR}/cuda-shim")
        file(MAKE_DIRECTORY "${_cuda_shim}/bin")
        if(NOT EXISTS "${_cuda_shim}/bin/nvcc")
            file(CREATE_LINK "${_nvcc}" "${_cuda_shim}/bin/nvcc" SYMBOLIC)
        endif()
        if(NOT EXISTS "${_cuda_shim}/include")
            if(EXISTS "/usr/include/cuda.h")
                file(CREATE_LINK "/usr/include" "${_cuda_shim}/include" SYMBOLIC)
            elseif(EXISTS "/usr/local/cuda/include")
                file(CREATE_LINK "/usr/local/cuda/include" "${_cuda_shim}/include" SYMBOLIC)
            endif()
        endif()
        set(_cuda "${_cuda_shim}")
        set(_nvcc "${_cuda}/bin/nvcc")
    else()
        message(FATAL_ERROR "Could not locate nvcc")
    endif()
    set(${out_var} "${_cuda}" PARENT_SCOPE)
    set(${nvcc_var} "${_nvcc}" PARENT_SCOPE)
endfunction()

function(modern_gpusim_detect_environment)
    set(options)
    set(oneValueArgs ACCELSIM_CONFIG)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_ACCELSIM_CONFIG)
        if(DEFINED ENV{ACCELSIM_CONFIG} AND NOT "$ENV{ACCELSIM_CONFIG}" STREQUAL "")
            set(ARG_ACCELSIM_CONFIG "$ENV{ACCELSIM_CONFIG}")
        else()
            set(ARG_ACCELSIM_CONFIG "release")
        endif()
    endif()

    if(DEFINED ENV{CUDA_INSTALL_PATH} AND NOT "$ENV{CUDA_INSTALL_PATH}" STREQUAL "")
        set(_cuda_in "$ENV{CUDA_INSTALL_PATH}")
    elseif(EXISTS "/usr/local/cuda")
        set(_cuda_in "/usr/local/cuda")
    else()
        set(_cuda_in "/usr")
    endif()

    _modern_gpusim_normalize_cuda_path("${_cuda_in}" _cuda _nvcc)

    execute_process(
        COMMAND ${_nvcc} --version
        OUTPUT_VARIABLE _nvcc_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "release [0-9]+\\.[0-9]+" _cuda_match "${_nvcc_out}")
    string(REGEX REPLACE "release " "" _cuda_ver "${_cuda_match}")
    string(REPLACE "." ";" _cuda_parts "${_cuda_ver}")
    list(GET _cuda_parts 0 _cuda_major)
    list(GET _cuda_parts 1 _cuda_minor)
    # Match gpgpu-sim version_detection.mk: printf("%02u%02u", 10*major, 10*minor)
    math(EXPR _cudart_version "${_cuda_major} * 1000 + ${_cuda_minor} * 10")

    if(CMAKE_CXX_COMPILER)
        set(_cxx_for_detect "${CMAKE_CXX_COMPILER}")
        if(IS_ABSOLUTE "${_cxx_for_detect}")
            get_filename_component(_cxx_bin_dir "${_cxx_for_detect}" DIRECTORY)
            string(REGEX REPLACE "g\\+\\+$" "gcc" _cc_for_ver "${_cxx_for_detect}")
        else()
            set(_cc_for_ver "gcc")
        endif()
    else()
        set(_cc_for_ver "gcc")
    endif()
    if(CMAKE_C_COMPILER AND IS_ABSOLUTE "${CMAKE_C_COMPILER}")
        set(_cc_for_ver "${CMAKE_C_COMPILER}")
        get_filename_component(_cc_bin_dir "${CMAKE_C_COMPILER}" DIRECTORY)
    endif()
    execute_process(
        COMMAND ${_cc_for_ver} --version
        OUTPUT_VARIABLE _gcc_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _cc_version "${_gcc_out}")
    if(NOT _cc_version)
        set(_cc_version "")
    endif()

    set(_gpgpusim_root "${CMAKE_CURRENT_LIST_DIR}/../gpu-simulator/gpgpu-sim")
    get_filename_component(_gpgpusim_root "${_gpgpusim_root}" ABSOLUTE)
    if(NOT EXISTS "${_gpgpusim_root}/Makefile")
        message(FATAL_ERROR "Bundled GPGPU-Sim not found at ${_gpgpusim_root}")
    endif()

    if(CMAKE_C_COMPILER AND IS_ABSOLUTE "${CMAKE_C_COMPILER}")
        set(_detect_cc "${CMAKE_C_COMPILER}")
    elseif(CMAKE_CXX_COMPILER AND IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
        string(REGEX REPLACE "g\\+\\+$" "gcc" _detect_cc "${CMAKE_CXX_COMPILER}")
    else()
        set(_detect_cc "gcc")
    endif()
    if(CMAKE_CXX_COMPILER AND IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
        set(_detect_cxx "${CMAKE_CXX_COMPILER}")
    else()
        set(_detect_cxx "g++")
    endif()
    if(CMAKE_C_COMPILER AND IS_ABSOLUTE "${CMAKE_C_COMPILER}")
        get_filename_component(_cc_bin_dir "${CMAKE_C_COMPILER}" DIRECTORY)
    elseif(CMAKE_CXX_COMPILER AND IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
        get_filename_component(_cc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    endif()
    if(CMAKE_CXX_COMPILER AND IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
        get_filename_component(_cxx_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    endif()

    execute_process(
        COMMAND /bin/bash -c
            "export CUDA_INSTALL_PATH=\"${_cuda}\" && \
             export CC=\"${_detect_cc}\" && \
             export CXX=\"${_detect_cxx}\" && \
             export PATH=\"${_cc_bin_dir}:${_cxx_bin_dir}:${_cuda}/bin:/usr/local/bin:/usr/bin:/bin\" && \
             source \"${_gpgpusim_root}/setup_environment\" \"${ARG_ACCELSIM_CONFIG}\" >/dev/null && \
             echo \"\${GPGPUSIM_CONFIG}\""
        OUTPUT_VARIABLE _gpgpusim_config_resolved
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_gpgpusim_config_resolved)
        set(_gpgpusim_config "${_gpgpusim_config_resolved}")
        set(_gpgpusim_lib_dir "${_gpgpusim_root}/lib/${_gpgpusim_config_resolved}")
    else()
        set(_gpgpusim_config "gcc-${_cc_version}/cuda-${_cudart_version}/${ARG_ACCELSIM_CONFIG}")
        set(_gpgpusim_lib_dir "${_gpgpusim_root}/lib/${_gpgpusim_config}")
    endif()

    set(CUDA_INSTALL_PATH "${_cuda}" PARENT_SCOPE)
    set(NVCC_EXECUTABLE "${_nvcc}" PARENT_SCOPE)
    set(CUDART_VERSION "${_cudart_version}" PARENT_SCOPE)
    set(CC_VERSION "${_cc_version}" PARENT_SCOPE)
    set(GPGPUSIM_ROOT "${_gpgpusim_root}" PARENT_SCOPE)
    set(GPGPUSIM_CONFIG "${_gpgpusim_config}" PARENT_SCOPE)
    set(GPGPUSIM_LIB_DIR "${_gpgpusim_lib_dir}" PARENT_SCOPE)
    set(ACCELSIM_CONFIG "${ARG_ACCELSIM_CONFIG}" PARENT_SCOPE)
endfunction()

function(modern_gpusim_write_version_header build_dir)
    execute_process(
        COMMAND git log --abbrev-commit -n 1
        COMMAND head -1
        COMMAND sed -re "s/commit (.*)/\\1/"
        OUTPUT_VARIABLE _git_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT _git_commit)
        set(_git_commit "unknown")
    endif()
    string(TIMESTAMP _time "%y-%m-%d-%H-%M-%S" UTC)
    set(_build "accelsim-commit-${_git_commit}_cmake_${_time}")
    file(WRITE "${build_dir}/accelsim_version.h"
        "const char *g_accelsim_version=\"${_build}\";\n")
endfunction()
