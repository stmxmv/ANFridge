//
// Created by aojoie on 4/2/2023.
//

#ifndef ANFRIDGE_INFERENCE_HPP
#define ANFRIDGE_INFERENCE_HPP

#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <string_view>
#include <span>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace ANFridge {

struct Detection {
    int class_id;
    float confidence;
    cv::Scalar color;
    cv::Rect box;
};

class Inference {
    bool cudaEnabled{};
    int _classNum;
    cv::Size2f modelShape{};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;

    float modelConfidenseThreshold{0.25};
    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};

    cv::Mat formatToSquare(const cv::Mat &source);

public:

    Inference(std::span<char> modelBytes,
              int classNum,
              const cv::Size &modelInputShape = {640, 640},
              bool runWithCuda = true);

    Inference(const char *onnxModelPath,
              int classNum,
              const cv::Size &modelInputShape = {640, 640},
              bool runWithCuda = true);

    std::vector<Detection> inference(const cv::Mat &input);

};


}// namespace ANFridge

#endif//ANFRIDGE_INFERENCE_HPP
