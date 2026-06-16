find_package(protobuf CONFIG QUIET)
if(protobuf_FOUND)
    set(_protobuf_lib protobuf::libprotobuf)
    set(_protobuf_protoc protobuf::protoc)
else()
    find_package(Protobuf REQUIRED)
    if(TARGET protobuf::libprotobuf)
        set(_protobuf_lib protobuf::libprotobuf)
    else()
        set(_protobuf_lib ${Protobuf_LIBRARIES})
    endif()
    if(TARGET protobuf::protoc)
        set(_protobuf_protoc protobuf::protoc)
    else()
        set(_protobuf_protoc ${Protobuf_PROTOC_EXECUTABLE})
    endif()
endif()

set(_proto_dir "${CMAKE_CURRENT_SOURCE_DIR}/dynamic_trace")
file(GLOB _proto_files "${_proto_dir}/*.proto")

set(_pb_srcs "")
foreach(_proto ${_proto_files})
    get_filename_component(_name "${_proto}" NAME_WE)
    list(APPEND _pb_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_name}.pb.cc")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_name}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${_name}.pb.h"
        COMMAND ${_protobuf_protoc}
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
            --proto_path=${_proto_dir}
            ${_proto}
        DEPENDS ${_proto}
        COMMENT "Generating protobuf ${_name}"
        VERBATIM
    )
endforeach()

file(GLOB _trace_srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
add_library(traces_enhanced STATIC ${_trace_srcs} ${_pb_srcs})
target_include_directories(traces_enhanced
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-simulator
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-simulator/ISA_Def
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-simulator/gpgpu-sim/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-simulator/trace-parser
)
if(TARGET protobuf::libprotobuf)
    target_link_libraries(traces_enhanced PUBLIC protobuf::libprotobuf)
else()
    target_include_directories(traces_enhanced PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(traces_enhanced PUBLIC ${_protobuf_lib})
endif()
set_target_properties(traces_enhanced PROPERTIES POSITION_INDEPENDENT_CODE ON)
