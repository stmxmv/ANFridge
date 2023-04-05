//
// Created by aojoie on 4/5/2023.
//

#ifndef ANFRIDGE_OCRDETECTOR_HPP
#define ANFRIDGE_OCRDETECTOR_HPP

#include <pd_inference_api.h>
#include <ANFridge/preprocess_op.h>
#include <ANFridge/postprocess_op.h>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <span>

namespace ANFridge {

class OCRDetector {

    PD_Config *config{};
    PD_Predictor *predictor{};

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
    bool is_scale_ = true;

    // pre-process
    PaddleOCR::ResizeImgType0 resize_op_;
    PaddleOCR::Normalize normalize_op_;
    PaddleOCR::Permute permute_op_;

    // post-process
    PaddleOCR::DBPostProcessor post_processor_;

public:

    ~OCRDetector();

    void loadModel(const char *model_dir);

    void run(cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes);
};


}


#endif//ANFRIDGE_OCRDETECTOR_HPP
