#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含此头文件以支持 __half 和 __half2

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

// 核心 Kernel：FP16x2 (half2) RCP 测试
// 只有使用 half2 才能触发 packed 指令 (Vectorization)
__global__ void rcp_half2_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 1. 初始化一组独立的 vector 寄存器变量
    __half2 val[ILP_FACTOR];
    
    // 给每个变量赋初值
    // 我们将 tid 和 i 转换为 float 后，打包成 half2
    // 逻辑：同样是 x = 1/x，数值在两个值之间震荡
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float init_v1 = (float)tid * 0.00001f + (float)i * 0.1f + 1.5f;
        float init_v2 = init_v1 + 0.05f; // 让向量的高位和低位稍微不同
        val[i] = __floats2half2_rn(init_v1, init_v2); 
    }

    // 2. 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用 FP16 向量化倒数指令
            // h2rcp 对应 PTX 指令: rcp.approx.ftz.f16x2
            // 这是一条指令同时计算两个半精度浮点数的倒数
            val[i] = h2rcp(val[i]);
        }
    }

    // 3. 归约结果
    // 为了防止编译器优化掉计算过程，我们需要聚合结果
    __half2 res_h2 = val[0];
    #pragma unroll
    for (int i = 1; i < ILP_FACTOR; i++) {
        res_h2 = __hadd2(res_h2, val[i]); // 使用向量加法聚合
    }
    
    // 4. 写回显存
    // 将 half2 拆解为两个 float 相加写入，方便 Host 端查看结果
    // (仅用于防优化，不计入性能核心路径)
    float final_res = __low2float(res_h2) + __high2float(res_h2);
    out[tid] = final_res;
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
    // FP16 吞吐量极高，需要大量 Warps 掩盖延迟
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数
    // 注意：这里计算的是“向量指令数 (Vector Instructions)”
    // 1个 rcp.f16x2 指令 = 2个浮点运算 (FLOPs)
    double inst_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    rcp_half2_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16x2 RCP)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // 增大 batch_size，因为 FP16 计算非常快，要避免 CPU Launch bound
        int batch_size = 50; 
        for(int i=0; i<batch_size; i++) {
            rcp_half2_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    double total_inst = inst_per_launch * total_launches;
    double ginst_per_sec = (total_inst / total_seconds) / 1e9;
    double gflops = ginst_per_sec * 2.0; // 乘以2，因为每个指令包含2个操作 (Packed)

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Vector Instructions: " << total_inst / 1e9 << " G Inst" << std::endl;
    std::cout << "FP16x2 RCP Instruction Throughput: " << ginst_per_sec << " G Inst/s" << std::endl;
    //std::cout << "FP16 RCP Operation Throughput:     " << gflops << " G Ops/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
