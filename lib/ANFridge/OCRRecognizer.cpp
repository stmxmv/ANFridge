//
// Created by aojoie on 4/5/2023.
//

#include "ANFridge/OCRRecognizer.hpp"

#include <format>

namespace ANFridge {

OCRRecognizer::~OCRRecognizer() {
    if (predictor) {
        PD_PredictorDestroy(predictor);
        /// i don't know but this will crash
//        PD_ConfigDestroy(config);
    }
}

void OCRRecognizer::loadModel(const char *model_dir, const char *labelListPath) {
    using namespace PaddleOCR;
    label_list_ = Utility::ReadDict(labelListPath);
    label_list_.insert(label_list_.begin(),
                       "#"); // blank char for ctc
    label_list_.push_back(" ");

    config = PD_ConfigCreate();
    PD_ConfigSetModel(config,
                      std::format("{}/inference.pdmodel", model_dir).c_str(),
                      std::format("{}/inference.pdiparams", model_dir).c_str());
    PD_ConfigDisableGpu(config);

    PD_ConfigSetCpuMathLibraryNumThreads(config, 4);

    PD_ConfigDeletePass(config, "matmul_transpose_reshape_fuse_pass");
    PD_ConfigSwitchIrOptim(config, true);
    PD_ConfigEnableMemoryOptim(config, true);

    predictor = PD_PredictorCreate(config);
}

void OCRRecognizer::run(std::span<cv::Mat> img_list,
                        vector<std::string> &rec_texts,
                        vector<float> &rec_text_scores) {

    using namespace PaddleOCR;

    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = Utility::argsort(width_list);

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale_ = true;
    bool use_tensorrt_ = false;
    std::string precision_ = "fp32";
    int rec_batch_num_ = 6;
    int rec_img_h_ = 32;
    int rec_img_w_ = 320;

    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};

    for (int beg_img_no = 0; beg_img_no < img_num;
         beg_img_no += rec_batch_num_) {
        int end_img_no = min(img_num, beg_img_no + rec_batch_num_);
        int batch_num = end_img_no - beg_img_no;
        int imgH = rec_image_shape_[1];
        int imgW = rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = max(max_wh_ratio, wh_ratio);
        }

        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            resize_op_.Run(srcimg, resize_img, max_wh_ratio,
                           use_tensorrt_, rec_image_shape_);
            normalize_op_.Run(&resize_img,mean_, scale_,
                              is_scale_);
            norm_img_batch.push_back(resize_img);
            batch_width = max(resize_img.cols, batch_width);
        }

        std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
        permute_op_.Run(norm_img_batch, input.data());
        // Inference.
        PD_OneDimArrayCstr* inputNames = PD_PredictorGetInputNames(predictor);
        PD_Tensor* input_t = PD_PredictorGetInputHandle(predictor, inputNames->data[0]);
        int shapes[] = { batch_num, 3, imgH, batch_width };
        PD_TensorReshape(input_t, std::size(shapes), shapes);
        //        input_t->Reshape({batch_num, 3, imgH, batch_width});
        PD_TensorCopyFromCpuFloat(input_t, input.data());
        //        input_t->CopyFromCpu(input.data());
        PD_PredictorRun(predictor);
        //        this->predictor_->Run();

        std::vector<float> predict_batch;
        PD_OneDimArrayCstr* outputNames = PD_PredictorGetOutputNames(predictor);
        PD_Tensor *output_t= PD_PredictorGetOutputHandle(predictor, outputNames->data[0]);
        //        auto output_names = this->predictor_->GetOutputNames();
        //        auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
        PD_OneDimArrayInt32 *predict_shape = PD_TensorGetShape(output_t);
        //        auto predict_shape = output_t->shape();

        int out_num = std::accumulate(predict_shape->data, predict_shape->data + predict_shape->size, 1,
                                      std::multiplies<>());
        predict_batch.resize(out_num + 1); // avoid win32 error report
        // predict_batch is the result of Last FC with softmax
        PD_TensorCopyToCpuFloat(output_t, predict_batch.data());
        //        output_t->CopyToCpu(predict_batch.data());

        // ctc decode
        for (int m = 0; m < predict_shape->data[0]; m++) {
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape->data[1]; n++) {
                // get idx
                argmax_idx = int(Utility::argmax(
                        &predict_batch[(m * predict_shape->data[1] + n) * predict_shape->data[2]],
                        &predict_batch[(m * predict_shape->data[1] + n + 1) * predict_shape->data[2]]));
                // get score
                max_value = float(*std::max_element(
                        &predict_batch[(m * predict_shape->data[1] + n) * predict_shape->data[2]],
                        &predict_batch[(m * predict_shape->data[1] + n + 1) * predict_shape->data[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list_[argmax_idx];
                    /// TODO label list ??
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (isnan(score)) {
                continue;
            }
            rec_texts[indices[beg_img_no + m]] = str_res;
            rec_text_scores[indices[beg_img_no + m]] = score;
        }

        PD_OneDimArrayCstrDestroy(inputNames);
        PD_OneDimArrayCstrDestroy(outputNames);
        PD_TensorDestroy(input_t);
        PD_TensorDestroy(output_t);
        PD_OneDimArrayInt32Destroy(predict_shape);
    }

}


}