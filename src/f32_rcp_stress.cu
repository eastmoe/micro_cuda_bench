#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度，用于隐藏指令延迟 (Instruction Level Parallelism)
// RCP (倒数) 指令通常延迟比普通 FADD/FMUL 高，也往往由 SFU 处理，需要 ILP 填充流水线
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

// 核心 Kernel：RCP (Reciprocal) 测试
__global__ void rcp_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 注意：避免 0.0，防止除零产生 INF (虽然 GPU 能处理，但为了数据稳健性)
    // 逻辑：x = 1/x，再次执行变为 x = 1/(1/x) = x，数值会在两个值之间震荡，不会溢出
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = (float)tid * 0.00001f + (float)i * 0.1f + 1.5f; 
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 rcp 之间没有依赖关系
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用内联 PTX 汇编强制生成 rcp.approx.ftz.f32 指令
            // 含义：approximate reciprocal (近似倒数), flush-to-zero (小树冲零)
            // 这是测试 GPU SFU (特殊函数单元) 吞吐能力的常用指令
            //asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(val[i]) : "f"(val[i]));
            
            // 如果你想测试标准除法 (1.0/x)，可以写成: val[i] = 1.0f / val[i];
            val[i] = 1.0f / val[i];
            // 但那通常受编译器优化选项(--use_fast_math)影响很大
        }
    }

    // 归约结果并写回，防止被编译器优化掉
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res += val[i];
    }
    
    // 只写入一次显存，减少显存带宽对计算测试的干扰
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
    // 为了跑满 GPU，我们需要足够的线程数
    int blockSize = 256;
    // RCP 指令通常吞吐量很高，需要足够的 active warps 掩盖延迟
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
    rcp_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (RCP)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // 每次发射一批 kernel，避免 CPU 开销过大
        int batch_size = 20; // 稍微增加 batch size，因为 rcp 可能比 erf 快
        for(int i=0; i<batch_size; i++) {
            rcp_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    std::cout << "FP32 RCP Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
