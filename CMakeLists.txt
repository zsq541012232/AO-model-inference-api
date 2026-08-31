cmake_minimum_required(VERSION 3.20)
project(model_infer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 配置路径
set(TENSORRT_ROOT "D:/AIforAO/TensorRT-8.6.1.6/TensorRT-8.6.1.6")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(OPENCV_ROOT "D:/opencv/build/x64/vc16")

# 头文件
include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    "D:/opencv/build/include"
)

# 库目录
link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
    ${OPENCV_ROOT}/lib
)

# 编译动态库DLL
add_library(model_infer SHARED
    tensorrt_infer.cpp
    model_dll.cpp
)

# 链接库
target_link_libraries(model_infer
    nvinfer
    cudart
    debug opencv_world480d
    optimized opencv_world480
)

# 测试程序
add_executable(api_tst api_tst.cpp)
target_link_libraries(api_tst
    model_infer
    debug opencv_world480d
    optimized opencv_world480
)

# 让DLL和EXE都输出至build/bin目录
set_target_properties(model_infer api_tst PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
