#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含，用于 __half 支持

// 宏定义：每个线程内部的展开程度
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

// 核心 Kernel：测试 FP16 MAD (HFMA) 吞吐量
// 使用 __half 类型，不使用 half2 (向量化) 和 Tensor Core
__global__ void mad_stress_kernel_fp16(__half* out, int n, __half add_val, __half mul_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量 (__half)
    __half val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 使用 __float2half 进行类型转换
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = __float2half((float)tid * 0.0001f + (float)i * 0.1f);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，利用指令级并行 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // FP16 乘加运算
            // 形式：val = val * mul + add
            // 编译器通常会将此处优化为 HFMA (Half-Precision Fused Multiply-Add) 指令
            // 严格遵守不使用内联函数(如 __hfma)和向量化(half2)的要求
            val[i] = val[i] * mul_val + add_val; 
        }
    }

    // 归约结果并写回，防止被编译器优化
    // 注意：FP16 累加容易溢出，但作为压力测试我们关注指令吞吐而非数值准确性
    __half res = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res = res + val[i];
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
    // FP16 单元通常需要大量线程来掩盖延迟
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存 (__half 类型)
    __half* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__half)));

    // 5. 计算每次 Kernel Launch 的总指令数
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 定义加数和乘数 (__half)
    // 保持乘数为 1.0 以避免快速溢出 (FP16 最大值仅为 65504)
    __half add_val = __float2half(0.001f);
    __half mul_val = __float2half(1.0f); 

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    mad_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, add_val, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 Scalar MAD)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        int batch_size = 50; // FP16 执行较快，增加 batch size 减少 launch 开销
        for(int i=0; i<batch_size; i++) {
            mad_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, add_val, mul_val);
        }
        total_launches += batch_size;

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        if (elapsed_ms >= duration_target * 1000.0) {
            break;
        }
    }

    // 8. 结果计算
    double total_seconds = elapsed_ms / 1000.0;
    double total_insts = ops_per_launch * total_launches; 
    double ginst_per_sec = (total_insts / total_seconds) / 1e9; 

    // 计算 TFLOPS 
    // 1个 HFMA 指令 = 2 FLOPs
    double tflops = (ginst_per_sec * 2.0) / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Instructions: " << total_insts / 1e9 << " G inst" << std::endl;
    std::cout << "FP16 (Scalar) MAD Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "Approx. FP16 Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Note: Without 'half2' (vectorization), performance on many GPUs" << std::endl;
    std::cout << "      (e.g., Ampere) may be limited to 1x FP32 throughput." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
