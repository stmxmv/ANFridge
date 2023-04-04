//
// HandlerState_startd by aojoie on 4/3/2023.
//

#include "fridge.grpc.pb.h"
#include "helloworld.grpc.pb.h"

#include <ANFridge/Inference.hpp>

#include <grpc++/grpc++.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <string>
#include <thread>


using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;


enum HandlerState {
    HandlerState_start,
    HandlerState_process,
    HandlerState_IOBreak, /// this could due to client write finish or connection lost
    HandlerState_willDestroy
};

class Handler {
    // The means of communication with the gRPC runtime for an asynchronous
    // server.
    grpc::Service *_service;
    // The producer-consumer queue where for asynchronous server notifications.
    ServerCompletionQueue *_completionQueue;
    // Context for the rpc, allowing to tweak aspects of it such as the use
    // of compression, authentication, as well as to send metadata back to the
    // client.
    ServerContext _serverContext;

    HandlerState _state;// The current serving state.

public:
    Handler(grpc::Service *service, ServerCompletionQueue *completionQueue)
        : _service(service), _completionQueue(completionQueue),
          _serverContext(), _state(HandlerState_start) {}

    template<typename T>
    bool isKindOf() {
        return typeid(*this) == typeid(T);
    }

    template<typename T>
        requires std::is_base_of_v<grpc::Service, T>
    T *getServicePtr() const { return dynamic_cast<T *>(_service); }

    template<typename T>
        requires std::is_base_of_v<grpc::Service, T>
    T &getService() const { return *dynamic_cast<T *>(_service); }

    grpc::Service *getServicePtr() const { return _service; }
    grpc::Service &getService() const { return *_service; }

    ServerCompletionQueue &getCompletionQueue() const { return *_completionQueue; }
    ServerCompletionQueue *getCompletionQueuePtr() const { return _completionQueue; }
    ServerContext &getServerContext() { return _serverContext; }
    ServerContext *getServerContextPtr() { return &_serverContext; }

    HandlerState getState() const { return _state; }
    void setState(HandlerState state) { _state = state; }

    virtual void dealloc() {
        delete this;
    }

    template<typename T, typename ...Args>
        requires std::is_base_of_v<Handler, T>
    static void SpawnAsyncHandler(Args &&...args) {
        T *handler = new T(std::forward<Args>(args)...);
        handler->resume();
    }

    virtual void resume() {}

};

class ObjectDetectImageHandler : public Handler {
    int times_ = 0;
    AN::ImageChunk imageChunk;
    ::grpc::ServerAsyncReader<AN::DetectResult, AN::ImageChunk> asyncReader;

    AN::DetectResult detectResult;

    ANFridge::Inference &inference;
    std::vector<uchar> imageData;

    void detectImageResumeInternal() {
        AN::ObjectDetector::AsyncService &service = getService<AN::ObjectDetector::AsyncService>();
        switch (getState()) {
            case HandlerState_start:
                setState(HandlerState_process);
                service.RequestDetectImage(getServerContextPtr(),
                                           &asyncReader,
                                           getCompletionQueuePtr(), getCompletionQueuePtr(), this);
                break;
            case HandlerState_process:
                if (times_ == 0) {
                    /// handle other client request
                    Handler::SpawnAsyncHandler<ObjectDetectImageHandler>(&service, getCompletionQueuePtr(), inference);
                }

                if (times_ > 0) {
                    imageData.insert(imageData.end(),
                                     imageChunk.buffer().begin(),
                                     imageChunk.buffer().end());
                }

                asyncReader.Read(&imageChunk, this);

                ++times_;
                break;
            case HandlerState_IOBreak: {
                setState(HandlerState_willDestroy);

                cv::Mat frame = cv::imdecode(imageData, cv::IMREAD_COLOR);

                if (frame.empty()) {
                    asyncReader.FinishWithError(Status(grpc::StatusCode::INVALID_ARGUMENT, "image data is empty"), this);
                    return;
                }

                std::vector<ANFridge::Detection> output = inference.inference(frame);

                for (const auto &d : output) {
                    AN::Detection *detection = detectResult.add_detection();
                    detection->set_x1(d.box.tl().x);
                    detection->set_y1(d.box.tl().y);
                    detection->set_x2(d.box.br().x);
                    detection->set_y2(d.box.br().y);
                    detection->set_confidence(d.confidence);
                    detection->set_id(d.class_id);
                }

                asyncReader.Finish(detectResult, Status::OK, this);
            } break;
            case HandlerState_willDestroy:
                GPR_ASSERT(getState() == HandlerState_willDestroy);
                dealloc();
                break;
        }
    }

public:
    ObjectDetectImageHandler(AN::ObjectDetector::AsyncService *service,
                             ServerCompletionQueue *cq,
                             ANFridge::Inference &inference)
        : Handler(service, cq), asyncReader(getServerContextPtr()), times_(0), inference(inference) {}

    virtual void resume() override {
        Handler::resume();
        detectImageResumeInternal();
    }
};


class ServerImpl {

    ANFridge::Inference inference;

    int _threadNum;

    AN::ObjectDetector::AsyncService service_;
    std::unique_ptr<Server> server_;

    std::atomic_bool running;
    std::vector<std::thread> handlerThreads;

    /// \AnyThread concurrent
    void handleRPCs(std::unique_ptr<ServerCompletionQueue> cq) {

        Handler::SpawnAsyncHandler<ObjectDetectImageHandler>(&service_, cq.get(), inference);

        void *tag;// uniquely identifies a request.
        bool ok = false;
        for (;;) {
            // Block waiting to read the next event from the completion queue.
            GPR_ASSERT(cq->Next(&tag, &ok));
            if (!running) {
                break;
            }
            Handler *handle = static_cast<Handler *>(tag);
            if (!ok) {
                if (handle->getState() == HandlerState_process) {
                    handle->setState(HandlerState_IOBreak);
                } else {
                    /// Got a canceled events, Maybe connection is closed unusually
                    handle->setState(HandlerState_willDestroy);
                }
            }

            handle->resume();
        }

        cq->Shutdown();
    }

public:
    explicit ServerImpl(int threadNum = 4)
        : inference("models/fridge.onnx", 28), _threadNum(threadNum) {}


    ~ServerImpl() {

        running = false;

        server_->Shutdown();
        // Always shutdown the completion queue after the server.

        for (auto &thread : handlerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }


    // There is no shutdown handling in this code.
    void run() {
        std::string server_address("0.0.0.0:50051");

        ServerBuilder builder;
        // Listen on the given address without any authentication mechanism.
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        // Register "service_" as the instance through which we'll communicate with
        // clients. In this case it corresponds to an *asynchronous* service.
        builder.RegisterService(&service_);
        // Get hold of the completion queue used for the asynchronous communication
        // with the gRPC runtime.


        // resume to the server's main loop.
        running = true;

        std::vector<std::unique_ptr<ServerCompletionQueue>> queues;
        for (int i = 0; i < _threadNum; ++i) {
            queues.emplace_back(builder.AddCompletionQueue());
        }

        // Finally assemble the server.
        server_ = builder.BuildAndStart();
        std::cout << "Server listening on " << server_address << std::endl;

        for (int i = 0; i < _threadNum; ++i) {
            handlerThreads.emplace_back([this, cq_ = std::move(queues[i])]() mutable {
                handleRPCs(std::move(cq_));
            });
        }

        /// block until other thread call Server::Shutdown
        server_->Wait();
    }
};

int main(int argc, char **argv) {
    for (;;) {
        try {
            ServerImpl server;
            server.run();
        } catch (const std::exception &exception) {
            std::cout << exception.what() << std::endl;
        }
    }

    return 0;
}