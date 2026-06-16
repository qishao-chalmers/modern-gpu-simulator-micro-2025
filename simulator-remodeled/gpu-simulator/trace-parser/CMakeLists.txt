add_library(trace-parser STATIC trace_parser.cc)
target_include_directories(trace-parser
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../ISA_Def
        ${GPGPUSIM_ROOT}/src
        ${GPGPUSIM_ROOT}/libcuda
        ${CUDA_INSTALL_PATH}/include
)
target_link_libraries(trace-parser PUBLIC traces_enhanced)
set_target_properties(trace-parser PROPERTIES POSITION_INDEPENDENT_CODE ON)
