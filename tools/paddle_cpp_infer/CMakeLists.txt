
aux_source_directory(src PADDLE_SRC)
add_an_tool(
        paddle_cpp_infer
        ${PADDLE_SRC}
)

target_link_libraries(paddle_cpp_infer PRIVATE ANFridge PaddleOCR)

target_include_directories(paddle_cpp_infer PRIVATE include ${CMAKE_CURRENT_LIST_DIR})