//
// Created by aojoie on 4/5/2023.
//

#ifndef ANFRIDGE_OCRRECOGNIZER_HPP
#define ANFRIDGE_OCRRECOGNIZER_HPP

#include <pd_inference_api.h>
#include <ANFridge/preprocess_op.h>
#include <ANFridge/postprocess_op.h>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <span>


namespace ANFridge {

class OCRRecognizer {
    // pre-process
    PaddleOCR::CrnnResizeImg resize_op_;
    PaddleOCR:: Normalize normalize_op_;
    PaddleOCR::PermuteBatch permute_op_;

    // post-process
    PaddleOCR::DBPostProcessor post_processor_;

    PD_Config *config{};
    PD_Predictor *predictor{};

    std::vector<std::string> label_list_;

    int rec_img_h_ = 32;
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale_ = true;
    bool use_tensorrt_ = false;
    std::string precision_ = "fp32";
    int rec_batch_num_ = 24;

public:

    ~OCRRecognizer();

    // Load Paddle inference model
    void loadModel(const char *model_dir, const char *labelListPath);

    void run(std::span<cv::Mat> img_list,
             std::vector<std::string> &rec_texts,
             std::vector<float> &rec_text_scores);

};

}


#endif//ANFRIDGE_OCRRECOGNIZER_HPP
