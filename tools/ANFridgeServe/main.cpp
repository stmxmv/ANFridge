//
// Created by aojoie on 4/3/2023.
//

#include "helloworld.grpc.pb.h"
#include "fridge.grpc.pb.h"

#include <ANFridge/Inference.hpp>

#include <grpc++/grpc++.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <string>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

class ObjectDetectorServiceImpl : public AN::ObjectDetector::Service {

    ANFridge::Inference inference;

public:
    explicit ObjectDetectorServiceImpl()
        : inference("models/fridge.onnx", 28) {}

    Status DetectImage(::grpc::ServerContext *context,
                       ::grpc::ServerReader<::AN::ImageChunk> *reader,
                       ::AN::DetectResult *response) override {
#ifdef AN_DEBUG
        std::cout << "DetectImage called" << std::endl;
#endif

        std::vector<uchar> imageData;

        AN::ImageChunk imageChunk;

        while (reader->Read(&imageChunk)) {
            imageData.insert(imageData.end(),
                             imageChunk.buffer().begin(),
                             imageChunk.buffer().end());
        }

        if (imageData.empty()) {
            return Status(grpc::StatusCode::INVALID_ARGUMENT, "image data is empty");
        }

        cv::Mat frame = cv::imdecode(imageData, cv::IMREAD_COLOR);

        if (frame.empty()) {
            return Status(grpc::StatusCode::INVALID_ARGUMENT, "image data is empty");
        }

        std::vector<ANFridge::Detection> output = inference.inference(frame);

        for (const auto &d : output) {
            AN::Detection *detection = response->add_detection();
            detection->set_x1(d.box.tl().x);
            detection->set_y1(d.box.tl().y);
            detection->set_x2(d.box.br().x);
            detection->set_y2(d.box.br().y);
            detection->set_confidence(d.confidence);
            detection->set_id(d.class_id);
        }

        return Status::OK;
    }
};

class GreeterServiceImpl final : public Greeter::Service {
    Status SayHello(ServerContext* context, const HelloRequest* request,
                    HelloReply* reply) override {
        reply->set_message("hello client");
        return Status::OK;
    }

    Status SayHelloAgain(ServerContext* context, const HelloRequest* request,
                         HelloReply* reply) override {
        std::string prefix("Hello again ");
        reply->set_message(prefix + request->name());
        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    GreeterServiceImpl service;
    ObjectDetectorServiceImpl objectDetectorService;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    builder.RegisterService(&objectDetectorService);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();

    return 0;
}