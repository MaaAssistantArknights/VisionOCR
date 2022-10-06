cmake_minimum_required(VERSION 3.16)
project(VisionOCR CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vnocr STATIC VisionOCR.mm)
add_executable(vnocr_test main.cpp)

find_package(OpenCV REQUIRED)

target_include_directories(vnocr PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vnocr
    ${OpenCV_LIBS}
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework Vision"
)

target_include_directories(vnocr_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vnocr_test
    vnocr
    ${OpenCV_LIBS}
    "-framework CoreFoundation"
    "-framework CoreGraphics"
    "-framework Vision"
)
