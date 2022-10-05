//
//  VisionOCR.h
//  VisionOCR
//
//  Created by hguandl on 30/9/2022.
//

#ifndef VisionOCR_h
#define VisionOCR_h

#include "ocr_exports.h"
#include <stddef.h>

struct paddle_ocr_t;
typedef int OCR_ERROR;
typedef unsigned char uint8_t;

#define OCR_SUCCESS 0
#define OCR_FAILURE 1

#ifdef __cplusplus
extern "C" {
#endif

OCRAPI_PORT paddle_ocr_t* OCR_CALL PaddleOcrCreate(const char* det_model_dir, const char* rec_model_dir,
                                                   const char* char_list_file, const char* cls_model_dir);

void OCRAPI PaddleOcrDestroy(paddle_ocr_t* ocr_ptr);

OCR_ERROR OCRAPI PaddleOcrRec(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
                              char** out_strs, float* out_scores, size_t* out_size,
                              double* out_times, size_t* out_times_size);

OCR_ERROR OCRAPI PaddleOcrSystem(paddle_ocr_t* ocr_ptr, const uint8_t* encode_buf, size_t encode_buf_size,
                                 bool with_cls,
                                 int* out_boxes, char** out_strs, float* out_scores, size_t* out_size,
                                 double* out_times, size_t* out_times_size);

#ifdef __cplusplus
}
#endif

#endif /* VisionOCR_h */
