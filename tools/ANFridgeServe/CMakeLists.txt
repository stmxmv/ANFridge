add_an_tool(
        ANFridgeServe
        main.cpp
)

target_link_libraries(ANFridgeServe PRIVATE ANFridge grpc proto)