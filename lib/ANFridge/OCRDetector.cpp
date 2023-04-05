//
// Created by aojoie on 4/5/2023.
//

#include "OCRDetector.hpp"

namespace ANFridge {

OCRDetector::~OCRDetector() {
    if (predictor) {
        PD_PredictorDestroy(predictor);
        /// i don't know but this will crash
        //    PD_ConfigDestroy(config);
    }
}

void OCRDetector::loadModel(const char *model_dir) {
    config = PD_ConfigCreate();
    PD_ConfigSetModel(config,
                      std::format("{}/inference.pdmodel", model_dir).c_str(),
                      std::format("{}/inference.pdiparams", model_dir).c_str());
    PD_ConfigDisableGpu(config);

    PD_ConfigSetCpuMathLibraryNumThreads(config, 4);

    PD_ConfigSwitchIrOptim(config, true);
    PD_ConfigEnableMemoryOptim(config, true);

    predictor = PD_PredictorCreate(config);
}

void OCRDetector::run(cv::Mat &img, vector<std::vector<std::vector<int>>> &boxes) {
    float ratio_h{};
    float ratio_w{};

    cv::Mat srcimg;
    cv::Mat resize_img;
    img.copyTo(srcimg);

    auto preprocess_start = std::chrono::steady_clock::now();
    this->resize_op_.Run(img, resize_img, this->limit_type_,
                         this->limit_side_len_, ratio_h, ratio_w,
                         this->use_tensorrt_);

    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                            this->is_scale_);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    this->permute_op_.Run(&resize_img, input.data());

    // Inference.
    PD_OneDimArrayCstr* inputNames = PD_PredictorGetInputNames(predictor);
    PD_Tensor* input_t = PD_PredictorGetInputHandle(predictor, inputNames->data[0]);
//    auto input_names = this->predictor_->GetInputNames();
//    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    int shapes[] = { 1, 3, resize_img.rows, resize_img.cols };
    PD_TensorReshape(input_t, std::size(shapes), shapes);
//    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    PD_TensorCopyFromCpuFloat(input_t, input.data());
//    input_t->CopyFromCpu(input.data());

    PD_PredictorRun(predictor);
//    this->predictor_->Run();

    std::vector<float> out_data;
    PD_OneDimArrayCstr* outputNames = PD_PredictorGetOutputNames(predictor);
    PD_Tensor *output_t= PD_PredictorGetOutputHandle(predictor, outputNames->data[0]);
//    auto output_names = this->predictor_->GetOutputNames();
//    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    PD_OneDimArrayInt32 *output_shape = PD_TensorGetShape(output_t);
//    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape->data, output_shape->data + output_shape->size, 1,
                                                    std::multiplies<>());

    out_data.resize(out_num + 1); /// // avoid win32 error report
    PD_TensorCopyToCpuFloat(output_t, out_data.data());
//    output_t->CopyToCpu(out_data.data());

    int n2 = output_shape->data[2];
    int n3 = output_shape->data[3];
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

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);

    PD_OneDimArrayCstrDestroy(inputNames);
    PD_OneDimArrayCstrDestroy(outputNames);
    PD_TensorDestroy(input_t);
    PD_TensorDestroy(output_t);
    PD_OneDimArrayInt32Destroy(output_shape);
}

}