#define MODEL_INFER_EXPORTS
#include "model_api.h"
#include "tensorrt_infer.h"
#include <cstring>
#include <algorithm>

void* Model_Init(const char* engine_path) {
    TensorRTInfer* infer = new TensorRTInfer();
    if (infer->loadEngine(engine_path)) {
        return static_cast<void*>(infer);
    }
    delete infer;
    return nullptr;
}

int Model_Infer(void* handle,
                const unsigned char* image_data,
                int width, int height, int channels,
                float* out_vector,
                int out_max_len) {
    if (!handle || !out_vector || out_max_len <= 0) return -1;
    TensorRTInfer* infer = static_cast<TensorRTInfer*>(handle);
    cv::Mat image(height, width, CV_8UC(channels), (void*)image_data);
    std::vector<float> pred_vec;
    bool ok = infer->infer(image, pred_vec);
    if (!ok) return -1;
    // 防止越界，只拷贝调用方能容纳的长度
    int copy_len = (std::min)(static_cast<int>(pred_vec.size()), out_max_len);
    memcpy(out_vector, pred_vec.data(), copy_len * sizeof(float));
    return copy_len;
}

void Model_Release(void* handle) {
    if (handle) {
        TensorRTInfer* infer = static_cast<TensorRTInfer*>(handle);
        delete infer;
    }
}
