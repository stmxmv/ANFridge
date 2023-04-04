add_library(grpc INTERFACE)

if (WIN32)
    set(GRPC_DIR "${CMAKE_SOURCE_DIR}/thirdparty/win32/grpc")
    set_target_properties(
            grpc PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GRPC_DIR}/include"
    )

    set(GRPC_LIB_PREFIX "${GRPC_DIR}/lib/debug")

    file(GLOB GRPC_LIBS_DEBUG "${GRPC_LIB_PREFIX}/*.lib")

    set(GRPC_LIB_PREFIX "${GRPC_DIR}/lib/release")

    file(GLOB GRPC_LIBS_RELEASE "${GRPC_LIB_PREFIX}/*.lib")

    target_link_libraries(
            grpc
            INTERFACE
            $<$<CONFIG:Debug>:${GRPC_LIBS_DEBUG}>
            $<$<CONFIG:Release>:${GRPC_LIBS_RELEASE}>
    )

else ()

    find_package(absl CONFIG REQUIRED) # https://github.com/grpc/grpc/issues/24976
    find_package(Protobuf REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
    if (NOT ${gRPC_FOUND})
        message(FATAL_ERROR "gRPC not found on this machine")
    endif ()

    target_link_libraries(
            grpc
            INTERFACE
            gRPC::grpc++
            gRPC::grpc++_reflection
    )

endif ()