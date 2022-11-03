//
//  main.cpp
//  VisionOCR
//
//  Created by hguandl on 30/9/2022.
//

#include <opencv2/opencv.hpp>

#include "VisionOCR.h"
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Please enter the filename and OCR profile" << std::endl;
        std::cerr << "Usage: << " << argv[0] << " <filename> PaddleOCR|PaddleCharOCR" << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(argv[1]);

    constexpr size_t MaxBoxSize = 256;
    constexpr size_t MaxTextSize = 4096;

    auto ocr = PaddleOcrCreate(argv[2], "", "", nullptr);

    int boxes_buffer[MaxBoxSize * 8] = {0};
    char* strs_buffer[MaxBoxSize] = {nullptr};
    float scores_buffer[MaxBoxSize] = {0};

    size_t size = 0;

    for (size_t i = 0; i != MaxBoxSize; ++i) {
        strs_buffer[i] = new char[MaxTextSize];
    }

    std::vector<uchar> buf;
    cv::imencode(".png", image, buf);

    auto result =
        PaddleOcrSystem(ocr, buf.data(), buf.size(), false, boxes_buffer,
                        strs_buffer, scores_buffer, &size, nullptr, nullptr);

    if (result == OCR_SUCCESS) {
        for (auto i = 0u; i < size; i++) {
            std::cout << i << "\t"
                      << "det boxes: ["
                      << "[" << boxes_buffer[i * 8] << ","
                      << boxes_buffer[i * 8 + 1] << "]"
                      << ","
                      << "[" << boxes_buffer[i * 8 + 2] << ","
                      << boxes_buffer[i * 8 + 3] << "]"
                      << ","
                      << "[" << boxes_buffer[i * 8 + 4] << ","
                      << boxes_buffer[i * 8 + 5] << "]"
                      << ","
                      << "[" << boxes_buffer[i * 8 + 6] << ","
                      << boxes_buffer[i * 8 + 7] << "]"
                      << "] rec text: " << strs_buffer[i]
                      << " rec score: " << scores_buffer[i] << std::endl;
        }
    } else {
        std::cerr << "OCR Failed." << std::endl;
    }

    for (size_t i = 0; i != MaxBoxSize; ++i) {
        delete[] strs_buffer[i];
    }

    PaddleOcrDestroy(ocr);

    return 0;
}
