//
//  VisionOCR.mm
//  VisionOCR
//
//  Created by hguandl on 30/9/2022.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#import <CoreFoundation/CoreFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Vision/Vision.h>

#include "VisionOCR.h"

struct paddle_ocr_t {};

// MARK: Utility functions

cv::Mat decode(const uint8_t* buffer, size_t size)
{
    std::vector<uint8_t> buf(buffer, buffer + size);
    return cv::imdecode(buf, 1);
}

CGImageRef CGImageFromCVMat(cv::Mat cvMat)
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGImageByteOrderDefault, //bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return imageRef;
}

NSError* recognize(CGImageRef image, NSMutableArray<VNRecognizedText*>* results) {
    auto handler = [[VNImageRequestHandler alloc]
                    initWithCGImage:image
                    options:@{}];

    auto group = dispatch_group_create();
    __block NSError* ocr_error = nil;

    auto request = [[VNRecognizeTextRequest alloc]
                    initWithCompletionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        if (error) {
            ocr_error = error;
            dispatch_group_leave(group);
            return;
        }

        if (request.results) {
            for (VNRecognizedTextObservation* observation in request.results) {
                [results addObject:[observation topCandidates:1][0]];
            }
        }
        dispatch_group_leave(group);
    }];
    request.recognitionLanguages = @[@"zh-Hans", @"en-US"];

    dispatch_group_enter(group);
    [handler performRequests:@[request] error: &ocr_error];
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

    return ocr_error;
}

// MARK: API implementations

paddle_ocr_t* PaddleOcrCreate(const char* det_model_dir, const char* rec_model_dir,
                              const char* char_list_file, const char* cls_model_dir)
{
    return new paddle_ocr_t();;
}

void PaddleOcrDestroy(paddle_ocr_t* ocr_ptr)
{
    if (ocr_ptr == nullptr) {
        return;
    }
    delete ocr_ptr;
    ocr_ptr = nullptr;
}

OCR_ERROR PaddleOcrRec(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
                       char** out_strs, float* out_scores, size_t* out_size,
                       double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || encode_buf == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    auto image = CGImageFromCVMat(srcimg);

    auto ocr_results = [[NSMutableArray<VNRecognizedText*> alloc] init];
    auto ocr_error = recognize(image, ocr_results);

    CGImageRelease(image);

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    auto count = ocr_results.count > 256 ? 256 : ocr_results.count;
    *out_size = count;

    for (auto i = 0u; i < count; i++) {
        auto candidate = ocr_results[i];
        out_strs[i] = const_cast<char*>([candidate.string cStringUsingEncoding:NSUTF8StringEncoding]);
        out_scores[i] = candidate.confidence;
    }

    [ocr_results release];

    return OCR_SUCCESS;
}

OCR_ERROR PaddleOcrSystem(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
                          bool with_cls,
                          int* out_boxes, char** out_strs, float* out_scores, size_t* out_size,
                          double* out_times, size_t* out_times_size)
{
    if (ocr_ptr == nullptr
        || encode_buf == nullptr
        || out_boxes == nullptr
        || out_strs == nullptr
        || out_scores == nullptr
        || out_size == nullptr) {
        return OCR_FAILURE;
    }

    cv::Mat srcimg = decode(encode_buf, encode_buf_size);
    if (srcimg.empty()) {
        return OCR_FAILURE;
    }

    auto image = CGImageFromCVMat(srcimg);

    auto ocr_results = [[NSMutableArray<VNRecognizedText*> alloc] init];
    auto ocr_error = recognize(image, ocr_results);

    CGImageRelease(image);

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    auto count = ocr_results.count > 256 ? 256 : ocr_results.count;
    *out_size = count;

    auto width = CGImageGetWidth(image);
    auto height = CGImageGetHeight(image);

    for (auto i = 0u; i < count; i++) {
        auto candidate = ocr_results[i];

        auto range = NSMakeRange(0, [[candidate string] length]);
        auto boxObservation = [candidate boundingBoxForRange:range error:&ocr_error];

        if (ocr_error) {
            break;
        }

        auto boundingBox = boxObservation ? boxObservation.boundingBox : CGRectZero;
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

        out_strs[i] = const_cast<char*>([[candidate string] UTF8String]);

        out_scores[i] = candidate.confidence;
    }

    [ocr_results release];

    if (ocr_error) {
        [ocr_error release];
        return OCR_FAILURE;
    }

    return OCR_SUCCESS;
}
