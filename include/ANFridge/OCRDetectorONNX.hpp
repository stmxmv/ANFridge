//
// Created by aojoie on 4/7/2023.
//

#ifndef ANFRIDGE_OCRDETECTORONNX_HPP
#define ANFRIDGE_OCRDETECTORONNX_HPP

#include <ANFridge/preprocess_op.h>
#include <ANFridge/postprocess_op.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <span>

namespace ANFridge {


class OCRDetectorONNX {

    Ort::Session session{ nullptr };
    Ort::SessionOptions sessionOptions;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;

    // pre-process
    PaddleOCR::ResizeImgType0 resize_op_;
    PaddleOCR::Normalize normalize_op_;
    PaddleOCR::Permute permute_op_;

    // post-process
    PaddleOCR::DBPostProcessor post_processor_;

    string limit_type_ = "max";
    int limit_side_len_ = 960;

    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    std::string det_db_score_mode_ = "slow";
    bool use_dilation_ = false;

    bool visualize_ = true;
    bool use_tensorrt_ = false;
    std::string precision_ = "fp32";

    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

    const float meanValues[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};

    bool is_scale_ = true;

    std::vector<int> _gpuIndices;

public:

    bool init(std::span<int> gpuIndices = {}) {
        _gpuIndices.assign(gpuIndices.begin(), gpuIndices.end());
        return true;
    }

    void loadModel(const char *modelPath);

    void run(const cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes);
};


}

#endif//ANFRIDGE_OCRDETECTORONNX_HPP
