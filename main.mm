//
//  main.mm
//  VisionOCR
//
//  Created by hguandl on 30/9/2022.
//

#include <opencv2/opencv.hpp>

#import <Foundation/Foundation.h>

#include "VisionOCR.h"
#include <iostream>

int main(int argc, const char * argv[]) {
    if (argc != 2) {
        std::cerr << "Please enter the filename" << std::endl;
        return 1;
    }

    @autoreleasepool {
        cv::Mat image = cv::imread(argv[1]);

        constexpr size_t MaxBoxSize = 256;

        auto ocr = PaddleOcrCreate("", "", "", nullptr);

        int boxes_buffer[MaxBoxSize * 8] = { 0 };
        char* strs_buffer[MaxBoxSize] = { nullptr };
        float scores_buffer[MaxBoxSize] = { 0 };

        size_t size = 0;

        std::vector<uchar> buf;
        cv::imencode(".png", image, buf);

        auto result = PaddleOcrSystem(ocr, buf.data(), buf.size(), false, boxes_buffer, strs_buffer, scores_buffer, &size, nil, nil);

        if (result == OCR_SUCCESS) {
            for (auto i = 0u; i < size; i++) {
                std::cout
                << i << "\t"
                << "det boxes: ["
                << "[" << boxes_buffer[i * 8] << "," << boxes_buffer[i * 8 + 1] << "]"
                << ","
                << "[" << boxes_buffer[i * 8 + 2] << "," << boxes_buffer[i * 8 + 3] << "]"
                << ","
                << "[" << boxes_buffer[i * 8 + 4] << "," << boxes_buffer[i * 8 + 5] << "]"
                << ","
                << "[" << boxes_buffer[i * 8 + 6] << "," << boxes_buffer[i * 8 + 7] << "]"
                << " rec text: " << strs_buffer[i]
                << " rec score: " << scores_buffer[i]
                << std::endl;
            }
        } else {
            std::cerr << "OCR Failed." << std::endl;
        }

        PaddleOcrDestroy(ocr);
    }
    return 0;
}
