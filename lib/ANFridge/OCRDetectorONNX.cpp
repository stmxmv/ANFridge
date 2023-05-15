//
// Created by aojoie on 4/7/2023.
//

#include "OCRDetectorONNX.hpp"

#include "utility.h"

namespace ANFridge {


void OCRDetectorONNX::loadModel(const char *modelPath) {
    static Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet");

    for (int index : _gpuIndices) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = index;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

//    OrtSessionOptionsAppendExecutionProvider_Dnnl(sessionOptions, true);

    sessionOptions.SetInterOpNumThreads(12);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    session = Ort::Session(env, AN::to_wstring(modelPath).c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath, sessionOptions);
#endif
    inputNamesPtr = AN::OrtSessionGetInputNames(&session);
    outputNamesPtr = AN::OrtSessionGetOutputNames(&session);
}

void OCRDetectorONNX::run(const cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes) {
    float ratio_h{};
    float ratio_w{};

    cv::Mat resize_img;

    this->resize_op_.Run(img, resize_img, this->limit_type_,
                         this->limit_side_len_, ratio_h, ratio_w,
                         this->use_tensorrt_);

    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                            this->is_scale_);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    this->permute_op_.Run(&resize_img, input.data());

    int64_t shapes[] = { 1, 3, resize_img.rows, resize_img.cols };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
                                                             input.data(), input.size(),
                                                             shapes, std::size(shapes));

    assert(inputTensor.IsTensor());
    const char * inputNames[] = { inputNamesPtr.data()->get() };
    const char * outputNames[] = { outputNamesPtr.data()->get() };
    auto outputTensor = session.Run(Ort::RunOptions{}, inputNames, &inputTensor,
                                              std::size(inputNames), outputNames, std::size(outputNames));

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1LL,
                                                       std::multiplies<>());

    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> out_data(floatArray, floatArray + outputCount);
    out_data.resize(outputCount + 1);

    int n2 = outputShape[2];
    int n3 = outputShape[3];
    int n = n2 * n3;

    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
        pred[i] = float(out_data[i]);
        cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (this->use_dilation_) {
        cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    boxes = post_processor_.BoxesFromBitmap(
            pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
            this->det_db_score_mode_);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, img);


}

}