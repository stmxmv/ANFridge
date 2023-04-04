add_an_tool(
        test-client
        main.cpp
        ../../proto/cpp/fridge.pb.cc
        ../../proto/cpp/fridge.grpc.pb.cc
        ../../proto/cpp/helloworld.pb.cc
        ../../proto/cpp/helloworld.grpc.pb.cc
)

target_link_libraries(test-client PRIVATE ANFridge grpc)