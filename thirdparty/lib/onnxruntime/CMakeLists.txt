add_library(onnxruntime INTERFACE)

if (WIN32)
    set(onnxruntime_DIR "${CMAKE_SOURCE_DIR}/thirdparty/win32/onnxruntime")
    set_target_properties(
            onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_DIR}/include"
    )

    target_link_libraries(
            onnxruntime
            INTERFACE
            ${onnxruntime_DIR}/lib/onnxruntime.lib
    )


else ()

    set(onnxruntime_DIR "${CMAKE_SOURCE_DIR}/thirdparty/linux/onnxruntime")
    set_target_properties(
            onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_DIR}/include"
    )

    target_link_libraries(
            onnxruntime
            INTERFACE
            ${onnxruntime_DIR}/lib/libonnxruntime.so
    )

endif ()