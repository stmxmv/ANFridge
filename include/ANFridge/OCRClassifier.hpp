//
// Created by aojoie on 4/5/2023.
//

#ifndef ANFRIDGE_OCRCLASSIFIER_HPP
#define ANFRIDGE_OCRCLASSIFIER_HPP

#include <pd_inference_api.h>
#include <ANFridge/preprocess_op.h>
#include <ANFridge/postprocess_op.h>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <span>


namespace ANFridge {

class OCRClassifier {

    PD_Config *config{};
    PD_Predictor *predictor{};

    bool use_gpu_ = false;
    int gpu_id_ = 0;
    int gpu_mem_ = 4000;
    int cpu_math_library_num_threads_ = 4;
    bool use_mkldnn_ = false;

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale_ = true;
    bool use_tensorrt_ = false;
    std::string precision_ = "fp32";
    int cls_batch_num_ = 20;
    // pre-process
    PaddleOCR::ClsResizeImg resize_op_;
    PaddleOCR::Normalize normalize_op_;
    PaddleOCR::PermuteBatch permute_op_;

public:

    double cls_thresh = 0.9;

    ~OCRClassifier();

    void loadModel(const char *model_dir);

    void run(std::vector<cv::Mat> img_list, std::vector<int> &cls_labels, std::vector<float> &cls_scores);
};

}

#endif//ANFRIDGE_OCRCLASSIFIER_HPP
