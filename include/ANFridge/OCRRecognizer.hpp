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
