//
// Created by aojoie on 4/3/2023.
//

#include "ANFridge/ObjectDetectorNative.h"
#include "ANFridge/Inference.hpp"

#include <opencv2/opencv.hpp>

using namespace ANFridge;

jlong Java_com_szu_refrigerator_nativelib_ObjectDetectorNative_initInternal
        (JNIEnv *env, jobject, jbyteArray modelBytes, jint classNum) {
    jsize size = env->GetArrayLength(modelBytes);
    jbyte* bufferPtr = env->GetByteArrayElements(modelBytes, nullptr);
//    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    Inference *inference = new Inference({ (char *)bufferPtr, (size_t)size }, classNum);
    return (jlong)inference;
}

void Java_com_szu_refrigerator_nativelib_ObjectDetectorNative_destroyInternal(JNIEnv *, jobject, jlong ptr) {
    Inference *inference = (Inference *)ptr;
    delete inference;
}

jobjectArray Java_com_szu_refrigerator_nativelib_ObjectDetectorNative_detectInternal
        (JNIEnv *env, jobject, jlong ptr, jbyteArray imageBytes) {
    Inference *inference = (Inference *)ptr;

    jsize size = env->GetArrayLength(imageBytes);
    jbyte* bufferPtr = env->GetByteArrayElements(imageBytes, nullptr);

    if (size == 0) {
        return nullptr;
    }

    std::vector<uchar> data(size);
    memcpy(data.data(), bufferPtr, size);

    cv::Mat frame = cv::imdecode(data, cv::IMREAD_COLOR);

    std::vector<Detection> output = inference->inference(frame);

    jclass detectionResultClass = env->FindClass("com/szu/refrigerator/nativelib/DetectionResult");

    if (!detectionResultClass) {
        printf("detectionResultClass not found\n");
        return nullptr;
    }

    jmethodID cid = env->GetMethodID(detectionResultClass, "<init>", "(FFFFFI)V");


    jobjectArray objectArray = env->NewObjectArray(output.size(), detectionResultClass, nullptr);

    for (int i = 0; i < output.size(); ++i) {
        float x1 = output[i].box.tl().x;
        float y1 = output[i].box.tl().y;
        float x2 = output[i].box.br().x;
        float y2 = output[i].box.br().y;
        env->SetObjectArrayElement(objectArray, i,
                                   env->NewObject(detectionResultClass, cid,
                                                  x1, y1, x2, y2, output[i].confidence, output[i].class_id));
    }

    return objectArray;
}
