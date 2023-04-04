add_an_library(
        ANFridge SHARED
        Inference.cpp
        ObjectDetectorNative.cpp
)

find_package(OpenCV REQUIRED)
find_package(JNI REQUIRED)

if (NOT ${OpenCV_FOUND})
    message(FATAL_ERROR "OpenCV not found on this machine")
endif ()

if (NOT ${JNI_FOUND})
    message(FATAL_ERROR "JNI not found on this machine")
endif ()

target_include_directories(ANFridge PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(ANFridge PUBLIC ${OpenCV_LIBS} PRIVATE JNI::JNI)

message(${OpenCV_LIBS})