add_an_tool(
        ANFridgeGUI
        WIN32
        main.cpp
)

add_library(ojoie INTERFACE)

target_include_directories(
        ojoie
        INTERFACE
        "C:/Users/aojoie/CLionProjects/ojoie/include"
        "C:/Users/aojoie/CLionProjects/ojoie/thirdparty/imgui/include"
        "C:/Users/aojoie/CLionProjects/ojoie/thirdparty/yaml/include"
        "C:/VulkanSDK/1.3.243.0/Include"
)

if (WIN32)
    set(
            OJOIE_LIBS_DEBUG
            "C:/Users/aojoie/CLionProjects/ojoie/cmake-build-ninja/lib/Debug/ojoied.lib"
    )

    set(
            OJOIE_LIBS
            "C:/Users/aojoie/CLionProjects/ojoie/cmake-build-ninja/lib/Release/ojoie.lib"
    )
    target_link_libraries(
            ojoie INTERFACE
            $<$<CONFIG:Debug>:${OJOIE_LIBS_DEBUG}>
            $<$<CONFIG:Release>:${OJOIE_LIBS}>
            $<$<CONFIG:RelWithDebInfo>:${OJOIE_LIBS}>
    )
endif()

add_library(libcurl INTERFACE)
target_include_directories(libcurl INTERFACE "C:/Users/aojoie/CLionProjects/thirdparty/vcpkg/installed/x64-windows/include")
target_link_libraries(libcurl INTERFACE "C:/Users/aojoie/CLionProjects/thirdparty/vcpkg/installed/x64-windows/lib/libcurl.lib")

target_link_libraries(ANFridgeGUI PRIVATE ANFridge ojoie libcurl)