#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含，用于 __half 和 __hrcp

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

// 核心 Kernel：FP16 scalar RCP 测试
// 注意：输入输出指针类型改为 __half
__global__ void hrcp_stress_kernel(__half* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的 FP16 寄存器变量
    __half val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 使用 __float2half 进行转换
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 初始值设为 1.5 左右，避免 0 或过大/过小的值
        // 逻辑：y = 1/x，再次 y = 1/y 变回 x，数值在两个值之间震荡
        float init_v = (float)tid * 0.00001f + (float)i * 0.1f + 1.5f;
        val[i] = __float2half(init_v);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 rcp 之间没有依赖关系
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 核心测试指令：__hrcp
            // 这会编译为 rcp.approx.f16 PTX 指令
            // 它是标量 FP16 的倒数指令
            val[i] = hrcp(val[i]);
        }
    }

    // 归约结果并写回，防止被编译器优化掉
    // 为了防止 FP16 累加溢出，这里也可以转换回 float 累加，
    // 但为了保持纯 FP16 寄存器压力，我们先用 __half 累加
    __half res = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res = __hadd(res, val[i]); // 使用 fp16 加法
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
    // 使用足够多的 Block 来填满 GPU SM
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    // 注意：这里分配的是 __half 类型的大小
    __half* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__half)));

    // 5. 计算每次 Kernel Launch 的总指令数 (Giga Instructions)
    // Total Ops = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    hrcp_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 Scalar __hrcp)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // FP16 吞吐通常很高，增加 batch size 减少 CPU launch overhead 占比
        int batch_size = 50; 
        for(int i=0; i<batch_size; i++) {
            hrcp_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    std::cout << "FP16 Scalar __hrcp Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
