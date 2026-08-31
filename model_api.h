#define MODEL_API __declspec(dllexport)
#else
#define MODEL_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
// 初始化模型，返回句柄
MODEL_API void* __cdecl Model_Init(const char* engine_path);
// 单张图像推理（纯C接口）
// image_data: 图像像素数据指针
// width/height/channels: 图像宽、高、通道数
// out_vector: 输出泽尼克系数
// out_max_len: 需要的泽尼克项数
// 返回值: 成功则返回实际输出的泽尼克项数，失败则返回-1
MODEL_API int __cdecl Model_Infer(void* handle,
                                  const unsigned char* image_data,
                                  int width, int height, int channels,
                                  float* out_vector,
                                  int out_max_len);
// 释放模型资源
MODEL_API void __cdecl Model_Release(void* handle);
#ifdef __cplusplus
}
#endif
