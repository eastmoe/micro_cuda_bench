#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 宏定义：每个线程内部的展开程度 (Instruction Level Parallelism)
// 增加此值可以隐藏指令延迟，确保存储器不是瓶颈
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

// 核心 Kernel：测试 FP16x2 Vectorized MAD (__hfma2) 吞吐量
// 使用 half2 类型，不使用 Tensor Core
__global__ void hfma2_stress_kernel(__half2* out, int n, __half scalar_add, __half scalar_mul) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 将标量输入转换为向量 (half2)，也就是把同一个值复制到高位和低位
    __half2 add_val = __half2half2(scalar_add);
    __half2 mul_val = __half2half2(scalar_mul);

    // 初始化一组独立的寄存器变量 (half2)
    __half2 val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值，模拟两个不同的 float 数据 packed 在一起
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float f1 = (float)tid * 0.0001f + (float)i * 0.1f;
        float f2 = f1 + 0.5f; 
        val[i] = __floats2half2_rn(f1, f2);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，利用指令级并行 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // FP16x2 向量乘加运算
            // 形式：val = val * mul + add (对两个 half 元素同时操作)
            // 这通常编译为 SASS 中的 HFMA2 指令
            val[i] = __hfma2(val[i], mul_val, add_val); 
        }
    }

    // 归约结果并写回，防止被编译器优化 (Dead Code Elimination)
    __half2 res = __float2half2_rn(0.0f);
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 使用向量加法累积
        res = __hadd2(res, val[i]);
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
    int numBlocks = props.multiProcessorCount * 64; 
    long long n_threads = (long long)numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    // 注意：这里 n 代表线程数，但每个线程处理的是 half2 (2个元素)
    std::cout << "Total Threads: " << n_threads << std::endl;

    // 4. 分配显存 
    // 输出数组大小为 n_threads 个 half2
    __half2* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n_threads * sizeof(__half2)));

    // 5. 计算每次 Kernel Launch 的总指令数
    // 每个线程执行 LOOP_ITER * ILP_FACTOR 条 __hfma2 指令
    double insts_per_launch = (double)n_threads * LOOP_ITER * ILP_FACTOR;
    
    // 定义加数和乘数 (__half)
    __half add_val = __float2half(0.001f);
    __half mul_val = __float2half(1.0f); 

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    hfma2_stress_kernel<<<numBlocks, blockSize>>>(d_out, n_threads, add_val, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16x2 __hfma2)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // FP16x2 吞吐量极高，增加 batch size 减少 launch overhead
        int batch_size = 100; 
        for(int i=0; i<batch_size; i++) {
            hfma2_stress_kernel<<<numBlocks, blockSize>>>(d_out, n_threads, add_val, mul_val);
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
    double total_insts = insts_per_launch * total_launches; 
    
    // GInst/s: 每秒执行的十亿条指令数
    double ginst_per_sec = (total_insts / total_seconds) / 1e9; 

    // 计算 TFLOPS 
    // 关键点：1个 __hfma2 指令 = 2个 FP16 元素 * (1乘 + 1加) = 4 FLOPs
    double tflops = (ginst_per_sec * 4.0) / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Instructions: " << total_insts / 1e9 << " G Insts (__hfma2)" << std::endl;
    std::cout << "Instruction Throughput: " << ginst_per_sec << " G Inst/s" << std::endl;
    std::cout << "FP16 (Packed) Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    //std::cout << "Note: Each __hfma2 instruction counts as 4 FLOPs." << std::endl;
    //std::cout << "      Requires Compute Capability >= 5.3 (best on Ampere/Hopper)." << std::endl;
    //std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
