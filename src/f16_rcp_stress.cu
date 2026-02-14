#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含此头文件以支持 half 类型

// 宏定义：每个线程内部的展开程度 (ILP)
#define ILP_FACTOR 8 

// 每次 Kernel 内部循环的次数
#define LOOP_ITER 1000

// 检查 CUDA 错误的辅助函数
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// 核心 Kernel：FP16 RCP (Reciprocal) 测试
// 输入输出指针类型改为 half*
__global__ void rcp_stress_kernel_fp16(half* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 1. 初始化一组独立的寄存器变量，使用 half 类型
    half val[ILP_FACTOR];
    
    // 给每个变量赋初值
    // 这里使用强制类型转换将 float 转为 half，不使用显式内联函数
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float init_val = (float)tid * 0.00001f + (float)i * 0.1f + 1.5f;
        val[i] = (half)init_val; 
    }

    // 2. 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，利用 ILP 掩盖延迟
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用内联 PTX 汇编强制生成 FP16 倒数指令: rcp.approx.f16
            // 约束 "h" 表示使用 16 位寄存器
            // 逻辑：x = 1 / x
            //asm("rcp.approx.f16 %0, %1;" : "=h"(val[i]) : "h"(val[i]));
            
            // 如果写成 val[i] = (half)1.0f / val[i]; 编译器可能会生成 div.f16 而不是 rcp，
            // 或者受 --use_fast_math 影响太大。
            val[i] = (half)1.0f / val[i];
        }
    }

    // 3. 归约结果
    // 累加结果以防止被编译器优化掉整个循环
    half res = (half)0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res = res + val[i]; // 使用 cuda_fp16.h 重载的 + 运算符
    }
    
    // 写入显存
    out[tid] = res;
}

int main(int argc, char* argv[]) {
    // 1. 参数解析
    double duration_target = 5.0; // 默认运行 5 秒
    if (argc > 1) {
        try {
            duration_target = std::stod(argv[1]);
        } catch (...) {
            std::cerr << "Usage: " << argv[0] << " [duration_in_seconds]" << std::endl;
            return 1;
        }
    }

    // 2. 获取设备信息
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;

    // 3. 配置 Grid 和 Block
    int blockSize = 256;
    // 保持足够的并发量
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存 (注意这里使用 sizeof(half))
    half* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(half)));

    // 5. 计算每次 Kernel Launch 的总指令数 (Giga Instructions)
    // Total Ops = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    rcp_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 RCP)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // FP16 通常比 FP32 快，可以适当增加 batch_size 以减少 CPU 发射开销的影响
        int batch_size = 50; 
        for(int i=0; i<batch_size; i++) {
            rcp_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n);
        }
        total_launches += batch_size;

        // 检查时间
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        if (elapsed_ms >= duration_target * 1000.0) {
            break;
        }
    }

    // 8. 结果计算
    double total_seconds = elapsed_ms / 1000.0;
    double total_ops = ops_per_launch * total_launches;
    double ginst_per_sec = (total_ops / total_seconds) / 1e9;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "FP16 RCP Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
