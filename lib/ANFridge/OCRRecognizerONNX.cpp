//
// Created by aojoie on 4/7/2023.
//
#include "OCRRecognizerONNX.hpp"
#include "utility.h"
#include <onnxruntime_c_api.h>
namespace ANFridge {


void OCRRecognizerONNX::loadModel(const char *modelPath,
                                  const char *labelListPath) {
    using namespace PaddleOCR;
    label_list_ = Utility::ReadDict(labelListPath);
    label_list_.insert(label_list_.begin(),
                       "#");// blank char for ctc
    label_list_.push_back(" ");

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
    inputNamesPtr  = AN::OrtSessionGetInputNames(&session);
    outputNamesPtr = AN::OrtSessionGetOutputNames(&session);
}

void OCRRecognizerONNX::run(std::span<cv::Mat> img_list, vector<std::string> &rec_texts, vector<float> &rec_text_scores) {

    using namespace PaddleOCR;

    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = Utility::argsort(width_list);


    for (int beg_img_no = 0; beg_img_no < img_num;
         beg_img_no += rec_batch_num_) {
        int end_img_no     = min(img_num, beg_img_no + rec_batch_num_);
        int batch_num      = end_img_no - beg_img_no;
        int imgH           = rec_image_shape_[1];
        int imgW           = rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h          = img_list[indices[ino]].rows;
            int w          = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio   = max(max_wh_ratio, wh_ratio);
        }

        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                           use_tensorrt_, rec_image_shape_);
            normalize_op_.Run(&resize_img, mean_, scale_,
                              is_scale_);
            norm_img_batch.push_back(resize_img);
            batch_width = max(resize_img.cols, batch_width);
        }

        std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
        permute_op_.Run(norm_img_batch, input.data());

        // Inference.
        int64_t shapes[]           = {batch_num, 3, imgH, batch_width};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor     = Ort::Value::CreateTensor<float>(memoryInfo,
                                                                 input.data(), input.size(),
                                                                 shapes, std::size(shapes));

        assert(inputTensor.IsTensor());
        const char *inputNames[]  = {inputNamesPtr.data()->get()};
        const char *outputNames[] = {outputNamesPtr.data()->get()};

        auto outputTensor = session.Run(Ort::RunOptions{},
                                        inputNames, &inputTensor,
                                        std::size(inputNames),
                                        outputNames, std::size(outputNames));

        assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());
        
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1LL,
                                                           std::multiplies<>());
        
        float *floatArray = outputTensor.front().GetTensorMutableData<float>();
        std::vector<float> predict_batch(floatArray, floatArray + outputCount);
        predict_batch.resize(outputCount + 1);// avoid win32 error report

        // ctc decode
        for (int m = 0; m < outputShape[0]; m++) {
            std::string str_res;
            int argmax_idx;
            int last_index  = 0;
            float score     = 0.f;
            int count       = 0;
            float max_value = 0.0f;

            for (int n = 0; n < outputShape[1]; n++) {
                // get idx
                argmax_idx = int(Utility::argmax(
                        &predict_batch[(m * outputShape[1] + n) * outputShape[2]],
                        &predict_batch[(m * outputShape[1] + n + 1) * outputShape[2]]));
                // get score
                max_value = float(*std::max_element(
                        &predict_batch[(m * outputShape[1] + n) * outputShape[2]],
                        &predict_batch[(m * outputShape[1] + n + 1) * outputShape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list_[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (isnan(score)) {
                continue;
            }
            rec_texts[indices[beg_img_no + m]]       = str_res;
            rec_text_scores[indices[beg_img_no + m]] = score;
        }
    }
}


}// namespace ANFridge