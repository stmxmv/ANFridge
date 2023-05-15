//
// Created by aojoie on 4/5/2023.
//

#include "OCR.hpp"

namespace ANFridge {

using namespace PaddleOCR;

OCR::OCR(const char *model_dir_det,
         const char *model_dir_cls,
         const char *model_dir_rec,
         const char *labelListPath,
         std::span<int> gpuIndices) {
    detector.init(gpuIndices);
    recognizer.init(gpuIndices);
    detector.loadModel(model_dir_det);
//    classifier.loadModel(model_dir_cls);
    recognizer.loadModel(model_dir_rec, labelListPath);
}

OCR::~OCR() {
}

std::vector<std::vector<PaddleOCR::OCRPredictResult>> OCR::ocr(std::span<cv::Mat> cv_all_imgs,
                                                                 bool det,
                                                                 bool rec,
                                                                 bool cls) {
    std::vector<std::vector<OCRPredictResult>> ocr_results;

    for (int i = 0; i < cv_all_imgs.size(); ++i) {
        std::vector<OCRPredictResult> ocr_result;

        cv::Mat srcimg = cv_all_imgs[i];
        if (!srcimg.data) {
            std::cerr << "[ERROR] image don't have any data" << endl;
            continue;
        }

        // det
        this->det(srcimg, ocr_result);

        // crop image
        std::vector<cv::Mat> img_list;
        if (ocr_result.empty()) {

            ocr_result.emplace_back();
            img_list.push_back(srcimg);

        } else {

            for (auto & j : ocr_result) {
                cv::Mat crop_img;
                crop_img = Utility::GetRotateCropImage(srcimg, j.box);
                img_list.push_back(crop_img);
            }
        }

//        this->cls(img_list, ocr_result);
//        for (int i = 0; i < img_list.size(); i++) {
//            if (ocr_result[i].cls_label % 2 == 1 &&
//                ocr_result[i].cls_score > classifier.cls_thresh) {
//                cv::rotate(img_list[i], img_list[i], 1);
//            }
//        }

        if (rec) {
            this->rec(img_list, ocr_result);
        }
        ocr_results.push_back(ocr_result);
    }

    return ocr_results;
}

void OCR::det(cv::Mat img, vector<PaddleOCR::OCRPredictResult> &ocr_results) {
    std::vector<std::vector<std::vector<int>>> boxes;
    detector.run(img, boxes);

    for (int i = 0; i < boxes.size(); i++) {
        OCRPredictResult res;
        res.box = boxes[i];
        ocr_results.push_back(res);
    }
    // sort boex from top to bottom, from left to right
    if (!ocr_results.empty())
        Utility::sorted_boxes(ocr_results);
}

void OCR::rec(std::vector<cv::Mat> img_list, vector<PaddleOCR::OCRPredictResult> &ocr_results) {
    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);
    std::vector<double> rec_times;
    recognizer.run(img_list, rec_texts, rec_text_scores);
    // output rec results
    for (int i = 0; i < rec_texts.size(); i++) {
        ocr_results[i].text  = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];
    }
}

void OCR::cls(const std::vector<cv::Mat>& img_list, vector<PaddleOCR::OCRPredictResult> &ocr_results) {
    throw std::runtime_error("not implemented");
//    std::vector<int> cls_labels(img_list.size(), 0);
//    std::vector<float> cls_scores(img_list.size(), 0);
//    std::vector<double> cls_times;
//
//    classifier.run(img_list, cls_labels, cls_scores);
//    // output cls results
//    for (int i = 0; i < cls_labels.size(); i++) {
//        ocr_results[i].cls_label = cls_labels[i];
//        ocr_results[i].cls_score = cls_scores[i];
//    }
}

void OCR::log(vector<double> &det_times, vector<double> &rec_times, vector<double> &cls_times, int img_num) {
}

}// namespace ANFridge