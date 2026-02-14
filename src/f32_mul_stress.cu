#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度
// F32 MUL (FMUL) 的延迟与 FADD 类似，ILP_FACTOR=8 足以掩盖延迟并填满流水线
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

// 核心 Kernel：测试 F32 MUL 吞吐量
// 参数 mul_val 用于防止编译器将循环折叠优化
__global__ void mul_stress_kernel(float* out, int n, float mul_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 注意：乘法测试要避免初值为 0，否则结果恒为 0
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = 1.0f + (float)tid * 0.00001f + (float)i * 0.01f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 mul 之间没有依赖关系 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // F32 乘法指令
            // 指令形式通常为：FMUL R, R, C (若 mul_val 在常量缓存中) 或 FMUL R, R, R
            val[i] = val[i] * mul_val; 
        }
    }

    // 归约结果并写回，防止被编译器视为死代码优化掉
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res += val[i];
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
    // 类似于 FADD，FMUL 的吞吐量极高，需要海量线程掩盖延迟
    int blockSize = 256;
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数 (Giga Instructions)
    // Total Ops = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 定义乘数
    // 选取一个非常接近 1.0 的数，防止在循环中数值迅速溢出到 INF 或下溢到 0
    // 虽然 GPU 算 INF 也是全速，但保持在正常数值范围内更稳妥
    float mul_val = 1.000001f;

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    mul_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (F32 MUL)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // 每次发射一批 kernel，减少 CPU Launch 开销的影响
        int batch_size = 20; 
        for(int i=0; i<batch_size; i++) {
            mul_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, mul_val);
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
    double ginst_per_sec = (total_ops / total_seconds) / 1e9; // G inst/s

    // 计算 TFLOPS (FP32 Mul 算 1 FLOP)
    // 注意：通常 GPU 标称 TFLOPS 是 FMA (2 Ops)，纯 MUL 理论峰值通常是 FMA 峰值的一半
    double tflops = ginst_per_sec / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "FP32 MUL Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "Approx. MUL Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
