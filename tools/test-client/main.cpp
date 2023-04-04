//
// Created by aojoie on 4/3/2023.
//

#include <grpc++/grpc++.h>
#include <string>

#include "fridge.grpc.pb.h"
#include "helloworld.pb.h"
#include "helloworld.grpc.pb.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientWriter;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

class GreeterClient {

    constexpr static size_t ChunkSize = 64 * 1024; // 64KB

public:
    GreeterClient(std::shared_ptr<Channel> channel)
        : _objectDetectorStub(AN::ObjectDetector::NewStub(channel)),
          stub_(Greeter::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string SayHello(const std::string& user) {
        // Data we are sending to the server.
        HelloRequest request;
        request.set_name(user);

        // Container for the data we expect from the server.
        HelloReply reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->SayHello(&context, request, &reply);

        // Act upon its status.
        if (status.ok()) {
            return reply.message();
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
            return "RPC failed";
        }
    }
    
    
    void detect(const char * imagePath) {
        std::ifstream imageFile(imagePath, std::ios::binary);
        if (!imageFile.is_open()) {
            return;
        }

        ClientContext context;
        AN::DetectResult detectResult;

        std::unique_ptr<ClientWriter<AN::ImageChunk>>
                writer(_objectDetectorStub->DetectImage(&context, &detectResult));

        AN::ImageChunk imageChunk;
        imageFile.unsetf(std::ios::skipws);

        std::vector<char> buffer(ChunkSize);
        while (!imageFile.eof()) {
            imageFile.read(buffer.data(), buffer.size());
            if (imageFile.gcount() > 0) {
                imageChunk.set_buffer(buffer.data(), (size_t)imageFile.gcount());
            }
            if (!writer->Write(imageChunk)) {
                break;
            }
        }

        writer->WritesDone();
        Status status = writer->Finish();

        if (status.ok()) {
            std::cout << detectResult.DebugString() << std::endl;
        }

    }

private:
    std::unique_ptr<AN::ObjectDetector::Stub> _objectDetectorStub;
    std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char** argv) {
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint specified by
    // the argument "--target=" which is the only expected argument.
    // We indicate that the channel isn't authenticated (use of
    // InsecureChannelCredentials()).
    std::string target_str;
    std::string arg_str("--target");
    if (argc > 1) {
        std::string arg_val = argv[1];
        size_t start_pos = arg_val.find(arg_str);
        if (start_pos != std::string::npos) {
            start_pos += arg_str.size();
            if (arg_val[start_pos] == '=') {
                target_str = arg_val.substr(start_pos + 1);
            } else {
                std::cout << "The only correct argument syntax is --target="
                          << std::endl;
                return 0;
            }
        } else {
            std::cout << "The only acceptable argument is --target=" << std::endl;
            return 0;
        }
    } else {
        target_str = "localhost:50051";
    }
    GreeterClient greeter(
            grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    std::string user("world");
    std::string reply = greeter.SayHello(user);

    greeter.detect("C:\\Users\\aojoie\\Desktop\\detect\\3.jpg");

    std::cout << "Greeter received: " << reply << std::endl;

    return 0;
}
