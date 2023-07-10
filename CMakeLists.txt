cmake_minimum_required(VERSION 3.22)
project(ANFridge VERSION 1.0 LANGUAGES CXX CUDA)

# /////////// compiler specific settings
if (MSVC)
    set(CMAKE_CXX_STANDARD 23)
    #[[$<$<CONFIG:Debug>:/analyze>]]
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus /await:strict /utf-8")
    # generate pdb file for release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    add_compile_definitions(NOMINMAX _UNICODE UNICODE)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-completeness")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-nullability-completeness")
endif()

message("debug flag: ${CMAKE_CXX_FLAGS_DEBUG}")
message("release with debug info flag: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("release flag: ${CMAKE_CXX_FLAGS_RELEASE}")
# /////////// compiler specific settings

# /////////// platform specific settings
set(OJOIE_USE_VULKAN TRUE)
if (APPLE)
    set(OJOIE_USE_VULKAN FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    # append third-party lib
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/thirdparty/macos)
    endif()
elseif(WIN32)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/thirdparty/win32)

elseif(UNIX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# /////////// platform specific settings

# /////////// project cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AN)

include_directories(proto/cpp)
include_directories(thirdparty/include)

# /////////// project cmake modules


option(VLD_MEM_CHECK "use vld to check memory" OFF)
if(VLD_MEM_CHECK)
    find_package(VLD)
    if(${VLD_FOUND})
        message("found vld at ${VLD_ROOT_DIR}")
    endif()
    add_compile_definitions($<$<CONFIG:Debug>:VLD_MEM_CHECK>)
endif()


add_library(ojoie INTERFACE)

target_include_directories(
        ojoie
        INTERFACE
        "C:/Users/aojoie/CLionProjects/ojoie/include"
        "C:/Users/aojoie/CLionProjects/ojoie/thirdparty/imgui/include"
        "C:/Users/aojoie/CLionProjects/ojoie/thirdparty/yaml/include"
        "C:/Users/aojoie/CLionProjects/ojoie/thirdparty/include"
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

add_subdirectory(thirdparty)
add_subdirectory(proto)
add_subdirectory(lib)
add_subdirectory(tools)


option(AN_BUILD_TEST "build tests" OFF)

if (AN_BUILD_TEST)
    message("Build tests is on")
    include(FetchContent)

    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    #     not install googletest
    set(INSTALL_GMOCK OFF CACHE BOOL "not install test" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "not install test" FORCE)

    add_subdirectory(test)

endif()



# finally include install module
include(install)
