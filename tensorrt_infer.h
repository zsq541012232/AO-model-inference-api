#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <cuda_runtime_api.h>

// TensorRT 日志类 (必须实现)
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 屏蔽大量无关信息刷屏 (kINFO和kVERBOSE)
        if (severity <= Severity::kWARNING)
            printf("[TensorRT] %s\n", msg);
    }
};

class TensorRTInfer {
public:
    TensorRTInfer();
    ~TensorRTInfer();
    // 加载引擎文件
    bool loadEngine(const std::string& engine_path);
    // 推理: 输入图像, 输出特征向量
    bool infer(const cv::Mat& image, std::vector<float>& zernike_vector);
    // 批量推理 (可选, 提升吞吐)
    bool inferBatch(const std::vector<cv::Mat>& images, std::vector<float>& features);
private:
    // 图像预处理
    void preprocess(const cv::Mat& image, float* input_buffer);
    Logger logger_;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<float> input_host_;
    void* buffers_[2] = {nullptr, nullptr}; // 0:输入GPU, 1: 输出GPU
    int input_size_ = 0;    // 输入元素总数
    int output_size_ = 0;   // 输出特征向量长度
    int input_h_ = 224;
    int input_w_ = 224;
    int input_idx_ = -1, output_idx_ = -1;   // 缓存binding索引
};
