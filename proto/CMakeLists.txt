macro(add_proto_src target)
    find_program(_PROTOBUF_PROTOC protoc)
    if(NOT _PROTOBUF_PROTOC)
        message(FATAL_ERROR "protoc not found!")
    else()
        message("Found protoc at ${_PROTOBUF_PROTOC}")
    endif()

    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    if (NOT _GRPC_CPP_PLUGIN_EXECUTABLE)
        message(FATAL_ERROR "grpc_cpp_plugin not found!")
    else()
        message("Found grpc_cpp_plugin at ${_GRPC_CPP_PLUGIN_EXECUTABLE}")
    endif()

    foreach(PROTO_SRC ${ARGN})

        get_filename_component(rg_proto ${PROTO_SRC} ABSOLUTE)
        get_filename_component(rg_proto_path "${rg_proto}" PATH)

        get_filename_component(PROTO_SRC_NAME ${PROTO_SRC} NAME_WE)

        # Generated sources
        set(rg_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_SRC_NAME}.pb.cc")
        set(rg_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_SRC_NAME}.pb.h")
        set(rg_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_SRC_NAME}.grpc.pb.cc")
        set(rg_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_SRC_NAME}.grpc.pb.h")

        add_custom_command(
                OUTPUT "${rg_proto_srcs}" "${rg_proto_hdrs}" "${rg_grpc_srcs}" "${rg_grpc_hdrs}"
                DEPENDS ${PROTO_SRC}
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                COMMAND ${_PROTOBUF_PROTOC}
                ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
                --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
                -I "${rg_proto_path}"
                --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
                "${rg_proto}"
                COMMENT "Compile proto source ${PROTO_SRC_NAME} ..."
        )
        list(APPEND PROTO_SRCS ${rg_proto_srcs})
        list(APPEND PROTO_SRCS ${rg_grpc_srcs})
        list(APPEND PROTO_HEADS ${rg_proto_hdrs})
        list(APPEND PROTO_HEADS ${rg_grpc_hdrs})
    endforeach()

    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_sources(${target} PRIVATE ${PROTO_SRCS} ${PROTO_HEADS})

endmacro()

file(GLOB PROTO_FILES *.proto)

add_library(proto OBJECT)

add_proto_src(proto ${PROTO_FILES})

target_include_directories(proto PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(proto PUBLIC grpc)