#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ANFridge/Inference.hpp>

#ifdef AN_DEBUG
#include <opencv2/core/utils/logger.hpp>
#endif

using namespace std;
using namespace cv;
using namespace ANFridge;

constexpr int class_num = 28;


#include <ANFridge/OCRRecognizer.hpp>
#include <ANFridge/OCR.hpp>

int main(int argc, char **argv) {

    using namespace PaddleOCR;

//    std::vector<cv::Mat> img_list;
//    cv::Mat img = cv::imread(argv[1]);
//    img_list.push_back(img);
//
//    std::vector<std::string> rec_texts(img_list.size());
//    std::vector<float> rec_text_scores(img_list.size());
//
//    ANFridge::OCRRecognizer recognizer;
//
//    recognizer.loadModel("./models/rec", "./models/ppocr_keys_v1.txt");
//
//    recognizer.run(img_list, rec_texts, rec_text_scores);
//
//    for (auto &text : rec_texts) {
//        std::cout << text << std::endl;
//    }

    int gpu[] = { 0 };
    ANFridge::OCR ocr("./models/det/det.onnx",
                      "./models/cls",
                      "./models/rec/rec.onnx",
                      "./models/ppocr_keys_v1.txt");

    std::vector<cv::String> imageNames;
    std::vector<cv::Mat> images;

    for (int i = 1; i < argc; ++i) {
        imageNames.emplace_back(argv[i]);
        images.emplace_back(cv::imread(argv[i]));
    }

    for (int i = 0; i < 10; ++i) {

        auto now = std::chrono::steady_clock::now();

        auto results = ocr.ocr(images);

        for (const auto &imageResult : results) {
            for (const auto &text : imageResult) {
                std::cout << text.text << std::endl;
            }
        }

        auto end      = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float>(end - now);

        std::cout << "================time: " << duration.count() << std::endl;
    }


    return 0;

    if (argc < 2) {
        return 1;
    }

#ifdef AN_DEBUG
//    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
#endif

//    bool runOnGPU = false;
//
//
//    Inference inf("./models/fridge.onnx", class_num, cv::Size(640, 640), runOnGPU);
//
//    std::vector<std::string> imageNames;
//    imageNames.emplace_back(argv[1]);
//
//    for (const auto &imageName : imageNames) {
//        cv::Mat frame = cv::imread(imageName);
//
//        // Inference starts here...
//        std::vector<Detection> output = inf.inference(frame);
//
//        int detections = output.size();
//        std::cout << "Number of detections:" << detections << std::endl;
//
//        for (int i = 0; i < detections; ++i) {
//            Detection detection = output[i];
//
//            cv::Rect box     = detection.box;
//            cv::Scalar color = detection.color;
//
//            // Detection box
//            cv::rectangle(frame, box, color, 2);
//
//            // Detection box text
//            std::string classString = std::to_string(detection.class_id) + ' ' + std::to_string(detection.confidence).substr(0, 4);
//            cv::Size textSize       = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
//            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
//
//            cv::rectangle(frame, textBox, color, cv::FILLED);
//            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
//        }
//        // Inference ends here...
//
//        // This is only for preview purposes
//        float scale = 0.8;
//
//        cv::namedWindow("Inference", cv::WINDOW_NORMAL);
//        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
//        cv::imshow("Inference", frame);
//
//        cv::waitKey(-1);
//    }
}