//
// Created by aojoie on 4/5/2023.
//

#ifndef ANFRIDGE_OCR_HPP
#define ANFRIDGE_OCR_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <ANFridge/preprocess_op.h>
#include <ANFridge/postprocess_op.h>

//#include <ANFridge/OCRDetector.hpp>
//#include <ANFridge/OCRRecognizer.hpp>
//#include <ANFridge/OCRClassifier.hpp>
#include <ANFridge/OCRDetectorONNX.hpp>
#include <ANFridge/OCRRecognizerONNX.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

namespace ANFridge {

class OCR {
    OCRDetectorONNX detector;
//    OCRDetector detector;
//    OCRClassifier classifier;
    OCRRecognizerONNX recognizer;
//    OCRRecognizer recognizer;

    void det(cv::Mat img, std::vector<PaddleOCR::OCRPredictResult> &ocr_results);

    void rec(std::vector<cv::Mat> img_list,
             std::vector<PaddleOCR::OCRPredictResult> &ocr_results);

    void cls(const std::vector<cv::Mat>& img_list,
             std::vector<PaddleOCR::OCRPredictResult> &ocr_results);
    void log(std::vector<double> &det_times, std::vector<double> &rec_times,
             std::vector<double> &cls_times, int img_num);

public:

    explicit OCR(const char *model_dir_det,
                 const char *model_dir_cls,
                 const char *model_dir_rec,
                 const char *labelListPath,
                 std::span<int> gpuIndices = {});

    ~OCR();

    std::vector<std::vector<PaddleOCR::OCRPredictResult>>
    ocr(std::span<cv::Mat> cv_all_img_names, bool det = true,
        bool rec = true, bool cls = true);

};

}

#endif//ANFRIDGE_OCR_HPP
