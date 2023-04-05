
aux_source_directory(${CMAKE_CURRENT_LIST_DIR} PROTO_SRCS)
add_library(
        proto OBJECT
        ${PROTO_SRCS}
)

target_include_directories(proto PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(proto PUBLIC grpc)