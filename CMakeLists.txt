cmake_minimum_required(VERSION 3.16)
include(FetchContent)

project(VisionOCR CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vnocr STATIC VisionOCR.mm)
add_executable(vnocr_test main.cpp)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  opencv
  URL https://github.com/MaaAssistantArknights/VisionOCR/releases/download/v0.1.0/opencv-4.5.3-macos-xcframework.zip
  URL_HASH SHA256=cab437694d0ef73568b4f401831ad6e86fbb1aa1cd2b618627f08626bc775b15
)
FetchContent_MakeAvailable(opencv)

set(CMAKE_FIND_FRAMEWORK ONLY)
find_library(OpenCV_Framework opencv2 REQUIRED
             PATHS ${opencv_SOURCE_DIR}/macos-arm64_x86_64)
get_filename_component(OpenCV_SEARCH_DIR ${OpenCV_Framework} DIRECTORY)
message(STATUS "Found opencv2.framework at: ${OpenCV_SEARCH_DIR}")

target_compile_options(vnocr PUBLIC -F${OpenCV_SEARCH_DIR})
target_compile_options(vnocr_test PRIVATE -F${OpenCV_SEARCH_DIR})

target_link_options(vnocr PUBLIC -F${OpenCV_SEARCH_DIR})
target_link_options(vnocr_test PRIVATE -F${OpenCV_SEARCH_DIR})

target_link_libraries(vnocr PUBLIC
    "-framework opencv2"
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework OpenCL"
    "-framework Vision"
)

target_link_libraries(vnocr_test
    vnocr
    "-framework opencv2"
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework OpenCL"
    "-framework Vision"
)
