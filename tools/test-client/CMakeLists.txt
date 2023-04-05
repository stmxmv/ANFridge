add_an_tool(
        test-client
        main.cpp
)

target_link_libraries(test-client PRIVATE ANFridge grpc proto)