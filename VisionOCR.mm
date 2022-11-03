//
//  VisionOCR.mm
//  VisionOCR
//
//  Created by hguandl on 30/9/2022.
//

#include <Foundation/Foundation.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#import <CoreGraphics/CoreGraphics.h>
#import <Vision/Vision.h>

#include "VisionOCR.h"

struct paddle_ocr_t {
    int profile;
};

// MARK: Utility functions

cv::Mat decode(const uint8_t* buffer, size_t size) {
    std::vector<uint8_t> buf(buffer, buffer + size);
    return cv::imdecode(buf, 1);
}

cv::Size scaled_size(cv::Size orig, int bound) {
    auto new_width = orig.width;
    auto new_height = orig.height;

    if (new_width < bound) {
        new_height = new_height * bound / new_width;
        new_width = bound;
    }

    if (new_height < bound) {
        new_width = new_width * bound / new_height;
        new_height = bound;
    }

    return cv::Size(new_width, new_height);
}

CGImageRef CGImageFromCVMat(cv::Mat cvMat) {
    NSData* data = [NSData dataWithBytes:cvMat.data
                                  length:cvMat.elemSize() * cvMat.total()];
    CGColorSpaceRef colorSpace;
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider =
        CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(
        cvMat.cols,                                   // width
        cvMat.rows,                                   // height
        8,                                            // bits per component
        8 * cvMat.elemSize(),                         // bits per pixel
        cvMat.step[0],                                // bytesPerRow
        colorSpace,                                   // colorspace
        kCGImageAlphaNone | kCGImageByteOrderDefault, // bitmap info
        provider,                                     // CGDataProviderRef
        NULL,                                         // decode
        false,                                        // should interpolate
        kCGRenderingIntentDefault                     // intent
    );
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return imageRef;
}

NSError* recognize(paddle_ocr_t* ocr_ptr, CGImageRef image,
                   NSMutableArray<VNRecognizedText*>* results) {
    auto handler = [[VNImageRequestHandler alloc] initWithCGImage:image
                                                          options:@{}];

    auto group = dispatch_group_create();
    __block NSError* ocr_error = nil;

    auto request = [[VNRecognizeTextRequest alloc]
        initWithCompletionHandler:^(VNRequest* _Nonnull request,
                                    NSError* _Nullable error) {
          if (error) {
              ocr_error = error;
              dispatch_group_leave(group);
              return;
          }

          if (request.results) {
              for (VNRecognizedTextObservation* observation in request
                       .results) {
                  [results addObject:[observation topCandidates:1][0]];
              }
          }
          dispatch_group_leave(group);
        }];
    request.revision = 2;
    request.recognitionLanguages = @[ @"zh-Hans", @"zh-Hant", @"en-US" ];

    if (ocr_ptr->profile == 1) {
        request.recognitionLevel = VNRequestTextRecognitionLevelFast;
    } else {
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    }

    dispatch_group_enter(group);
    [handler performRequests:@[ request ] error:&ocr_error];
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

    [handler release];
    [request release];

    return ocr_error;
}

NSError* recognize(paddle_ocr_t* ocr_ptr, cv::Mat srcimg,
                   NSMutableArray<VNRecognizedText*>* results) {
    auto size = srcimg.size();
    CGImageRef image;
    if (size.width < 50 || size.height < 50) {
        cv::Mat dstimg;
        auto new_size = scaled_size(size, 50);
        cv::resize(srcimg, dstimg, new_size, 0, 0, cv::INTER_CUBIC);
        image = CGImageFromCVMat(dstimg);
    } else {
        image = CGImageFromCVMat(srcimg);
    }

    auto error = recognize(ocr_ptr, image, results);
    CGImageRelease(image);
    return error;
}

// MARK: API implementations

paddle_ocr_t* PaddleOcrCreate(const char* det_model_dir,
                              const char* rec_model_dir __unused,
                              const char* char_list_file __unused,
                              const char* cls_model_dir __unused) {
    auto model_name = [NSString stringWithUTF8String:det_model_dir];
    if ([model_name containsString:@"PaddleOCR"]) {
        return new paddle_ocr_t{0};
    } else if ([model_name containsString:@"PaddleCharOCR"]) {
        return new paddle_ocr_t{1};
    } else {
        return nullptr;
    }
}

void PaddleOcrDestroy(paddle_ocr_t* ocr_ptr) {
    if (ocr_ptr == nullptr) {
        return;
    }
    delete ocr_ptr;
    ocr_ptr = nullptr;
}

OCR_ERROR _PaddleOcrRec(paddle_ocr_t* ocr_ptr, cv::Mat srcimg, char** out_strs,
                        float* out_scores, size_t* out_size) {
    if (srcimg.empty() || out_strs == nullptr || out_scores == nullptr ||
        out_size == nullptr) {
        return OCR_FAILURE;
    }

    auto ocr_results = [[NSMutableArray<VNRecognizedText*> alloc] init];
    auto ocr_error = recognize(ocr_ptr, srcimg, ocr_results);

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    auto count = ocr_results.count > 256 ? 256 : ocr_results.count;
    *out_size = count;

    for (auto i = 0u; i < count; i++) {
        auto candidate = ocr_results[i];
        strncpy(out_strs[i], [[candidate string] UTF8String], 4095);
        out_scores[i] = candidate.confidence;
    }

    [ocr_results release];

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrRec(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf,
                       size_t encode_buf_size, char** out_strs,
                       float* out_scores, size_t* out_size,
                       double* out_times __unused,
                       size_t* out_times_size __unused) {
    if (ocr_ptr == nullptr || encode_buf == nullptr || out_strs == nullptr ||
        out_scores == nullptr || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrRec(ocr_ptr, srcimg, out_strs, out_scores, out_size);
}

OCR_ERROR OCRAPI PaddleOcrRecWithData(paddle_ocr_t* ocr_ptr, int rows, int cols,
                                      int type, void* data, char** out_strs,
                                      float* out_scores, size_t* out_size,
                                      double* out_times __unused,
                                      size_t* out_times_size __unused) {
    if (ocr_ptr == nullptr || data == nullptr || out_strs == nullptr ||
        out_scores == nullptr || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg(rows, cols, type, data);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrRec(ocr_ptr, srcimg, out_strs, out_scores, out_size);
}

OCR_ERROR _PaddleOcrSystem(paddle_ocr_t* ocr_ptr, cv::Mat srcimg,
                           int* out_boxes, char** out_strs, float* out_scores,
                           size_t* out_size) {
    if (srcimg.empty() || out_boxes == nullptr || out_strs == nullptr ||
        out_scores == nullptr || out_size == nullptr) {
        return OCR_FAILURE;
    }

    auto ocr_results = [[NSMutableArray<VNRecognizedText*> alloc] init];
    auto ocr_error = recognize(ocr_ptr, srcimg, ocr_results);

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    auto count = ocr_results.count > 256 ? 256 : ocr_results.count;
    *out_size = count;

    auto width = srcimg.size().width;
    auto height = srcimg.size().height;

    for (auto i = 0u; i < count; i++) {
        auto candidate = ocr_results[i];

        auto range = NSMakeRange(0, [[candidate string] length]);
        auto boxObservation = [candidate boundingBoxForRange:range
                                                       error:&ocr_error];

        if (ocr_error) {
            break;
        }

        auto boundingBox =
            boxObservation ? boxObservation.boundingBox : CGRectZero;
        auto boxRect = VNImageRectForNormalizedRect(boundingBox, width, height);

        // Top left
        out_boxes[i * 8] = boxRect.origin.x;
        out_boxes[i * 8 + 1] = height - boxRect.origin.y - boxRect.size.height;
        // Top right
        out_boxes[i * 8 + 2] = boxRect.origin.x + boxRect.size.width;
        out_boxes[i * 8 + 3] = height - boxRect.origin.y - boxRect.size.height;
        // Bottom right
        out_boxes[i * 8 + 4] = boxRect.origin.x + boxRect.size.width;
        out_boxes[i * 8 + 5] = height - boxRect.origin.y;
        // Bottom left
        out_boxes[i * 8 + 6] = boxRect.origin.x;
        out_boxes[i * 8 + 7] = height - boxRect.origin.y;

        strncpy(out_strs[i], [[candidate string] UTF8String], 4095);

        out_scores[i] = candidate.confidence;
    }

    [ocr_results release];

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrSystem(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf,
                          size_t encode_buf_size, bool with_cls __unused,
                          int* out_boxes, char** out_strs, float* out_scores,
                          size_t* out_size, double* out_times __unused,
                          size_t* out_times_size __unused) {
    if (ocr_ptr == nullptr || encode_buf == nullptr || out_boxes == nullptr ||
        out_strs == nullptr || out_scores == nullptr || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrSystem(ocr_ptr, srcimg, out_boxes, out_strs, out_scores,
                            out_size);
}

OCR_ERROR OCRAPI PaddleOcrSystemWithData(paddle_ocr_t* ocr_ptr, int rows,
                                         int cols, int type, void* data,
                                         bool with_cls __unused, int* out_boxes,
                                         char** out_strs, float* out_scores,
                                         size_t* out_size,
                                         double* out_times __unused,
                                         size_t* out_times_size __unused) {
    if (ocr_ptr == nullptr || data == nullptr || out_boxes == nullptr ||
        out_strs == nullptr || out_scores == nullptr || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg(rows, cols, type, data);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    return _PaddleOcrSystem(ocr_ptr, srcimg, out_boxes, out_strs, out_scores,
                            out_size);
}
