add_library(ffmpeg INTERFACE)

target_include_directories(ffmpeg INTERFACE include)

target_link_libraries(
        ffmpeg INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/lib/avcodec.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/avdevice.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/avfilter.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/avformat.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/avutil.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/postproc.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/swresample.lib
        ${CMAKE_CURRENT_LIST_DIR}/lib/swscale.lib
)