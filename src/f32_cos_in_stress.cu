#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度，用于隐藏指令延迟 (Instruction Level Parallelism)
// SFU (Special Function Unit) 虽然吞吐量通常比 FMA 低，但也需要指令级并行来跑满流水线
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

// 核心 Kernel：测试 __cosf (Fast Cosine Intrinsic)
__global__ void fast_cos_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 注意：__cosf 在某些架构上对输入范围有要求（通常是大数值），
    // 但因为我们下面做的是迭代计算 val = cos(val)，结果总是在 [-1, 1] 之间，所以输入范围是安全的。
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = (float)tid * 0.0001f + (float)i * 0.1f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 __cosf 之间没有依赖关系
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用 __cosf (硬件固有指令 Intrinsic)
            // 这通常会被编译为单条 MUFU.COS 指令
            val[i] = __cosf(val[i]); 
        }
    }

    // 归约结果并写回，防止被编译器优化掉整个计算过程
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

    // 3. 配置 Grid 和 Block
    // SFU 单元的数量通常少于 CUDA Core (FP32 单元)，需要足够的线程来确保持续压力
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
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    fast_cos_stress_kernel<<<numBlocks, blockSize>>>(d_out, n); // 调用 fast cos kernel
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test for __cosf..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // 每次发射一批 kernel，避免 CPU 开销过大
        int batch_size = 10; 
        for(int i=0; i<batch_size; i++) {
            fast_cos_stress_kernel<<<numBlocks, blockSize>>>(d_out, n); // 调用 fast cos kernel
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
    std::cout << "Test Type: Fast Math Intrinsic (__cosf)" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "FP32 __cosf Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
