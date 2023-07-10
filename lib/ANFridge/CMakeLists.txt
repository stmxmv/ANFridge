add_an_library(
        ANFridge SHARED
        Inference.cpp
#        ObjectDetectorNative.cpp
#        OCRRecognizer.cpp
#        OCRDetector.cpp
#        OCRClassifier.cpp
        OCR.cpp
        OCRDetectorONNX.cpp
        OCRRecognizerONNX.cpp


        clipper.cpp
        postprocess_op.cpp
        preprocess_op.cpp
        utility.cpp
)

find_package(OpenCV REQUIRED)
find_package(JNI REQUIRED)

if (NOT ${OpenCV_FOUND})
    message(FATAL_ERROR "OpenCV not found on this machine")
endif ()

if (NOT ${JNI_FOUND})
    message(FATAL_ERROR "JNI not found on this machine")
endif ()

target_include_directories(ANFridge PUBLIC ${OpenCV_INCLUDE_DIRS} PRIVATE ${CMAKE_SOURCE_DIR}/include/ANFridge)

target_link_libraries(ANFridge PUBLIC opencv_world onnxruntime ojoie)

message(${OpenCV_LIBS})