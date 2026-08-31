#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "model_api.h"
// 字符串分割工具（用于解析CSV）
std::vector<std::string> splitStr(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}
// 读取泽尼克系数CSV (默认一行逗号分隔格式，支持任意项数)
bool readZernikeCSV(const std::string& csv_path, std::vector<float>& out_coeffs) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "[error]fail to open CSV: " << csv_path << std::endl;
        return false;
    }
    auto parse = [&out_coeffs](const std::string& s) -> bool {
        try { out_coeffs.push_back(std::stof(s)); return true;}
        catch (...) {return false;}
    };
    out_coeffs.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.find(',') != std::string::npos) {
            for (const auto& tok : splitStr(line, ',')) {
                if (!parse(tok)) {
                    std::cerr << "[error] fail to parse CSV token: " << tok << std::endl;
                    return false;
                }
            }
        } else {
            std::istringstream iss(line);
            std::string tok;
            while (iss >> tok) {
                if (!parse(tok)) {
                    std::cerr << "[error] fail to parse CSV token: " << tok << std::endl;
                    return false;
                }
            }
        }
    }
    if (out_coeffs.empty()) {
        std::cerr << "[error] CSV is empty: " << csv_path << std::endl;
        return false;
    }
    return true;
}
// 计算预测值与真值的MAE、RMSE
void calcMetric(const std::vector<float>& pred,
                const std::vector<float>& gt,
                float& mae, float& rmse) {
    size_t n = std::min(pred.size(), gt.size());
    if (0 == n) {
        mae = -1.f;
        rmse = -1.f;
        return;
    }
    float sum_abs = 0.f, sum_sq = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float diff = pred[i] - gt[i];
        sum_abs += std::abs(diff);
        sum_sq += diff * diff;
    }
    mae = sum_abs / n;
    rmse = std::sqrt(sum_sq / n);
}
int main() {
    // ==================== 配置参数 ==========================
    const std::string data_root = "D:/AIforAO/deploy/test-dataset/";   // 测试数据根目录
    const std::string engine_path = "D:/AIforAO/deploy/model/model_fp16_4000.engine";  // 引擎文件路径
    const int start_id = 1;        // 样本起始编号
    const int end_id = 2500;       // 样本结束编号
    const int id_pad_digit = 0;    // 编号补零位数，如4对应imgIF0001.jpg, 0对应imgIF1.jpg
    const int input_channels = 2;  // 模型输入通道数
    const int zernike_nums = 15;   // 泽尼克项数
    const bool save_detail = true; // 是否保存每个样本的详细结果
    const std::string result_file = "test_result.csv";
    // =======================================================
    // 1. 初始化模型
    std::cout << ">>> loading model..." << std::endl;
    void* model = Model_Init(engine_path.c_str());
    if (!model)
    {
        std::cerr << "fail to load model, engine path and packages path may error." << std::endl;
        system("pause");
        return -1;
    }
    std::cout << ">>> model loaded, start testing." << std::endl;
    // 2. 统计量初始化
    int total = end_id - start_id + 1;
    int success = 0, fail = 0;
    float total_mae = 0.f, total_rmse = 0.f;
    float max_mae = 0.f;
    int max_mae_id = -1;
    // 结果文件
    std::ofstream fout;
    if (save_detail) {
        fout.open(result_file);
        fout << "ID,MAE,RMSE" << std::endl;
    }
    // 复用变量，避免频繁内存申请
    cv::Mat img_if, img_podf;
    cv::Mat input_mat;
    std::vector<float> zernike_gt;
    std::vector<float> pred_vec;
    auto t_start = std::chrono::high_resolution_clock::now();
    // 3. 批量遍历测试
    for (int id = start_id; id <= end_id; ++id) {
        // 生成带补零的编号字符串
        std::ostringstream ss;
        ss << std::setw(id_pad_digit) << std::setfill('0') << id;
        std::string id_str = ss.str();
        // 拼接文件路径
        std::string path_if = data_root + "imgIF" + id_str + ".jpg";
        std::string path_podf = data_root + "imgPoDF" + id_str + ".jpg";
        std::string path_csv = data_root + "Zernike" + id_str + ".csv";
        // 3.1 读取单通道图像
        img_if = cv::imread(path_if, cv::IMREAD_GRAYSCALE);
        img_podf = cv::imread(path_podf, cv::IMREAD_GRAYSCALE);
        if (img_if.empty() || img_podf.empty()) {
            std::cerr << "[warning] faile to read No." << id << " images, pass" << std::endl;
            fail++;
            continue;
        }
        // 3.2 读取泽尼克真值
        if (!readZernikeCSV(path_csv, zernike_gt)) {
            fail++;
            std::cerr << "[warning] Fail to read No." << id << " Zernike coe, Pass" << std::endl;
            continue;
        }
        // 3.3 两个单通道图像合并(HWC排布)
        // 注意：通道顺序必须与训练时完全一致，顺序相反请调换两个Mat
        std::vector<cv::Mat> ch_list = { img_if, img_podf };
        cv::merge(ch_list, input_mat);

        // 3.4 调用DLL推理
        pred_vec.resize(zernike_nums);
        int ok = Model_Infer(
            model,
            input_mat.data,
            input_mat.cols,
            input_mat.rows,
            input_channels,
            pred_vec.data(),
            zernike_nums
        );
        if (-1 == ok || pred_vec.empty()) {
            std::cerr << "[warning] Fail to inference No." << id << " example, Pass" << std::endl;
            fail++;
            continue;
        }
        // 3.5 精度校验
        float mae, rmse;
        calcMetric(pred_vec, zernike_gt, mae, rmse);
        if (mae < 0) {
            std::cerr << "[warning] No." << id << " output is not match gt, pass" << std::endl;
            fail++;
            continue;
        }
        // 3.6 更新统计
        success++;
        total_mae += mae;
        total_rmse += rmse;
        if (mae > max_mae) {
            max_mae = mae;
            max_mae_id = id;
        }
        // 保存单样本结果
        if (save_detail) {
            fout << id << ","
                 << std::fixed << std::setprecision(8) << mae << ","
                 << rmse << std::endl;
        }
        // 打印进度 (每100组输出一次)
        if (id % 100 == 0 || id == end_id) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double cost = std::chrono::duration<double>(t_now - t_start).count();
            std::cout << "progress: " << id << "/" <<total
                      << " | success: " << success
                      << " | mean MAE: " << std::fixed << std::setprecision(6) << total_mae / success
                      << " | time cost: " << cost << "s" << std::endl;
        }
    }
    // 4. 输出最终统计结果
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n====================== test report ====================" << std::endl;
    std::cout << "Total Examples Number: " << total << std::endl;
    std::cout << "Success: " << success << " | Fail: " << fail << std::endl;
    std::cout << "Total time cost: " << total_time << "s" << std::endl;
    std::cout << "single inference time cost " << total_time / success * 1000 << " ms" << std::endl;
    std::cout << "golbal mean MAE: " << std::fixed << std::setprecision(8) << total_mae / success << std::endl;
    std::cout << "global mean RMSE: " << total_rmse / success << std::endl;
    std::cout << "Max MAE: " << max_mae << "(Example No.:" << max_mae_id << ")" << std::endl;
    std::cout << "======================================================" << std::endl;
    // 5. 资源释放
    if (save_detail) {
        fout.close();
        std::cout << "Detailed result is saved to : " << result_file << std::endl;
    }
    Model_Release(model);
    system("pause");
    return 0;
}
