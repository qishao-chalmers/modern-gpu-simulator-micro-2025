add_library(trace-driven STATIC trace_driven.cc)
target_include_directories(trace-driven
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../ISA_Def
        ${GPGPUSIM_ROOT}/src
        ${GPGPUSIM_ROOT}/libcuda
        ${CUDA_INSTALL_PATH}/include
)
target_link_libraries(trace-driven PUBLIC traces_enhanced)
set_target_properties(trace-driven PROPERTIES POSITION_INDEPENDENT_CODE ON)
