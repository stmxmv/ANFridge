//
// Created by aojoie on 4/5/2023.
//

#include "OCRClassifier.hpp"

namespace ANFridge {

using namespace PaddleOCR;

OCRClassifier::~OCRClassifier() {
    if (predictor) {
        PD_PredictorDestroy(predictor);
        /// i don't know but this will crash
        //    PD_ConfigDestroy(config);
    }
}
void OCRClassifier::loadModel(const char *model_dir) {
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

void OCRClassifier::run(std::vector<cv::Mat> img_list, vector<int> &cls_labels, vector<float> &cls_scores) {
    int img_num = img_list.size();
    std::vector<int> cls_image_shape = {3, 48, 192};
    for (int beg_img_no = 0; beg_img_no < img_num;
         beg_img_no += this->cls_batch_num_) {
        auto preprocess_start = std::chrono::steady_clock::now();
        int end_img_no = min(img_num, beg_img_no + this->cls_batch_num_);
        int batch_num = end_img_no - beg_img_no;
        // preprocess
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[ino].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, this->use_tensorrt_,
                                 cls_image_shape);

            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                                    this->is_scale_);
            norm_img_batch.push_back(resize_img);
        }
        std::vector<float> input(batch_num * cls_image_shape[0] *
                                         cls_image_shape[1] * cls_image_shape[2],
                                 0.0f);
        this->permute_op_.Run(norm_img_batch, input.data());

        // inference.
        PD_OneDimArrayCstr* inputNames = PD_PredictorGetInputNames(predictor);
        PD_Tensor* input_t = PD_PredictorGetInputHandle(predictor, inputNames->data[0]);
//        auto input_names = this->predictor_->GetInputNames();
//        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        int shapes[] = { batch_num, cls_image_shape[0], cls_image_shape[1], cls_image_shape[2] };
        PD_TensorReshape(input_t, std::size(shapes), shapes);
//        input_t->Reshape({batch_num, cls_image_shape[0], cls_image_shape[1],
//                          cls_image_shape[2]});
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
        predict_batch.resize(out_num + 1);

        PD_TensorCopyToCpuFloat(output_t, predict_batch.data());
//        output_t->CopyToCpu(predict_batch.data());

        // postprocess
        for (int batch_idx = 0; batch_idx < predict_shape->data[0]; batch_idx++) {
            int label = int(
                    Utility::argmax(&predict_batch[batch_idx * predict_shape->data[1]],
                                    &predict_batch[(batch_idx + 1) * predict_shape->data[1]]));
            float score = float(*std::max_element(
                    &predict_batch[batch_idx * predict_shape->data[1]],
                    &predict_batch[(batch_idx + 1) * predict_shape->data[1]]));
            cls_labels[beg_img_no + batch_idx] = label;
            cls_scores[beg_img_no + batch_idx] = score;
        }

        PD_OneDimArrayCstrDestroy(inputNames);
        PD_OneDimArrayCstrDestroy(outputNames);
        PD_TensorDestroy(input_t);
        PD_TensorDestroy(output_t);
        PD_OneDimArrayInt32Destroy(predict_shape);

    }
}

}