add_library(PaddleOCR INTERFACE)

if (WIN32)
    set(PaddleOCR_DIR "${CMAKE_SOURCE_DIR}/thirdparty/win32/PaddleOCR")
    set_target_properties(
            PaddleOCR PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PaddleOCR_DIR}/paddle/include"
    )

    target_link_libraries(
            PaddleOCR
            INTERFACE
            ${PaddleOCR_DIR}/paddle/lib/paddle_inference_c.lib
    )

#    target_include_directories(
#            PaddleOCR
#            INTERFACE
#            ${PADDLE_LIB}/third_party/install/protobuf/include
#            ${PADDLE_LIB}/third_party/install/glog/include
#    )

else ()

    set(PaddleOCR_DIR "${CMAKE_SOURCE_DIR}/thirdparty/linux/PaddleOCR")
    set_target_properties(
            PaddleOCR PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PaddleOCR_DIR}/paddle/include"
    )

    target_link_libraries(
            PaddleOCR
            INTERFACE
            ${PaddleOCR_DIR}/paddle/lib/paddle_inference_c.a
    )


endif ()