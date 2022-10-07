cmake_minimum_required(VERSION 3.16)
project(VisionOCR CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vnocr STATIC VisionOCR.mm)
add_executable(vnocr_test main.cpp)

set(CMAKE_FIND_FRAMEWORK ONLY)
find_library(OpenCV_Framework opencv2 REQUIRED)
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
