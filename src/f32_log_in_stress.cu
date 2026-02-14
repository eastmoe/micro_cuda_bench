#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度，用于隐藏指令延迟 (Instruction Level Parallelism)
// SFU (Special Function Unit) 延迟较高，且数量通常少于 FP32 Core (CUDA Cores)
// 因此通常需要较高的 ILP 才能测出峰值吞吐
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

// 核心 Kernel：测试 __logf (Intrinsic Fast Log)
__global__ void fast_log_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 这里故意使用正数开始
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 避免 0，防止直接 -inf，给一个合适的正数范围
        val[i] = (float)tid * 0.00001f + (float)(i + 1) * 0.5f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 __logf 之间没有依赖关系
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // ==========================================================
            // 修改点：使用 __logf (CUDA Intrinsic)
            // 这是一个近似计算，直接编译为 GPU SFU 指令 (通常是 lg2.approx)
            // ==========================================================
            val[i] = __logf(val[i]); 
            
            // 注意：__logf 对输入范围比较敏感。
            // 连续做 log 很快会变成负数 -> NaN。
            // 对于单纯测试指令发射吞吐量（Instruction Throughput），
            // 即使是处理 NaN，SFU 的流水线占用通常是一样的。
            // 如果想要更有意义的数值计算测试，可以使用 val[i] = __logf(fabsf(val[i]) + 1.0f);
            // 但为了保持和原程序 pure instruction latency 的对比，这里保持原样。
        }
    }

    // 归约结果并写回，防止被编译器优化掉
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res += val[i];
    }
    
    // 只写入一次显存
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
    std::cout << "Test Target: __logf (Fast Intrinsic Log)" << std::endl;

    // 3. 配置 Grid 和 Block
    // SFU 单元通常比 CUDA Core 少（例如 Ampere 架构 FP32:SFU = 16:1 或 32:1）
    // 所以由于资源竞争，可能不需要极其巨大的并发量就能跑满 SFU，
    // 但保持高并发有助于隐藏延迟。
    int blockSize = 256;
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数 (Giga Instructions)
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    fast_log_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // 每次发射一批 kernel，避免 CPU 开销过大
        int batch_size = 20; // 稍微增加 batch，SFU kernel 可能执行得比 FP32 慢（带宽低）或快（指令少），视架构而定
        for(int i=0; i<batch_size; i++) {
            fast_log_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    std::cout << "Intrinsic __logf Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
