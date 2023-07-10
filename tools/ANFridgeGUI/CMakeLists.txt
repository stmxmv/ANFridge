add_an_tool(
        ANFridgeGUI
        WIN32
        main.cpp
        resources.rc
        .manifest
)

add_an_library(ANFridge_CUDA STATIC util.cu)
set_target_properties(
        ANFridge_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 11
)

# drag and drop not work when run at user mode
#set_property(
#        TARGET ANFridgeGUI PROPERTY
#        LINK_FLAGS "/MANIFESTUAC:level='requireAdministrator'"
#)

add_library(libcurl INTERFACE)
target_include_directories(libcurl INTERFACE "C:/Users/aojoie/CLionProjects/thirdparty/vcpkg/installed/x64-windows/include")
target_link_libraries(libcurl INTERFACE "C:/Users/aojoie/CLionProjects/thirdparty/vcpkg/installed/x64-windows/lib/libcurl.lib")

target_link_libraries(ANFridgeGUI PRIVATE ANFridge ojoie)
target_link_libraries(ANFridgeGUI PRIVATE ANFridge ojoie libcurl ffmpeg)