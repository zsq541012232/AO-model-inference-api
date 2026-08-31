#include <fstream>
#include <cstring>
#include "tensorrt_infer.h"
// ========== 预处理和训练完全一致，先归一化到[0,1]，再做log1p处理 ==========
const bool NORMALIZE = true;         // 是否除以255
const bool USE_LOG_PREPROCESS = true; // 是否做log1p处理
// ==================================================================

TensorRTInfer::TensorRTInfer() {}
TensorRTInfer::~TensorRTInfer() {
    // 释放资源
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (buffers_[0]) cudaFree(buffers_[0]);
    if (buffers_[1]) cudaFree(buffers_[1]);
    if (stream_) cudaStreamDestroy(stream_);
}

bool TensorRTInfer::loadEngine(const std::string& engine_path) {
    // 读取engine文件到内存
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    // 反序列化引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) return false;
    context_ = engine_->createExecutionContext();
    if (!context_) return false;
    // 获取输入输出维度
    int input_idx = engine_->getBindingIndex("input");
    int output_idx = engine_->getBindingIndex("output");
    if (input_idx < 0 || output_idx < 0) {
        printf("[TensorRT] binding name not found! check engine tensor names\n");
        return false;
    }
    nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_idx);
    nvinfer1::Dims output_dims = engine_->getBindingDimensions(output_idx);
    printf("[TensorRT] input dims: [%d, %d, %d, %d], output dims: [%d, %d]\n",
        input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3],
        output_dims.d[0], output_dims.d[1]);
    // NCHW
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
    input_size_ = 1 * input_dims.d[1] * input_h_ * input_w_;
    output_size_ = 1 * output_dims.d[1];  // 泽尼克系数长度
    // 分配GPU显存
    cudaMalloc(&buffers_[input_idx], input_size_ * sizeof(float));
    cudaMalloc(&buffers_[output_idx], output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);
    return true;
}

void TensorRTInfer::preprocess(const cv::Mat& image, float* input_buffer) {
    cv::Mat input_img;
    cv::resize(image, input_img, cv::Size(input_w_, input_h_));
    input_img.convertTo(input_img, CV_32FC2);
    // 归一化，除以255
    if (NORMALIZE) input_img = input_img / 255.0f;
    // 做log1p预处理
    if (USE_LOG_PREPROCESS) {
        input_img.forEach<cv::Vec2f>([](cv::Vec2f& pixel, const int*) -> void {
            pixel[0] = std::log1p(pixel[0]);
            pixel[1] = std::log1p(pixel[1]);
        });
    }
    std::vector<cv::Mat> channels(2);
    cv::split(input_img, channels);
    // 按CHW顺序填充
    int idx = 0;
    for (int c = 0; c < 2; c++) { // OpenCV是BGR，反转成RGB
        for (int h = 0; h < input_h_; h++) {
            for (int w = 0; w < input_w_; w++) {
                input_buffer[idx++] = channels[c].at<float>(h, w);
            }
        }
    }
}

bool TensorRTInfer::infer(const cv::Mat& image, std::vector<float>& zernike_vector) {
    if (!context_) return false;
    // 1. 预处理
    std::vector<float> input_host(input_size_);
    preprocess(image, input_host.data());
    // 2. 数据拷贝: CPU -> GPU
    int input_idx = engine_->getBindingIndex("input");
    int output_idx = engine_->getBindingIndex("output");
    cudaMemcpyAsync(buffers_[input_idx], input_host.data(),
                    input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    // 3. 执行推理
    context_->enqueueV2(buffers_, stream_, nullptr);
    // 4. 结果拷贝: GPU -> CPU
    zernike_vector.resize(output_size_);
    cudaMemcpyAsync(zernike_vector.data(), buffers_[output_idx],
                    output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // 5. 同步流
    cudaStreamSynchronize(stream_);
    return true;
}
