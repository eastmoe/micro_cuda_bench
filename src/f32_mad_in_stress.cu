#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度
// 现代 GPU 需要足够的指令级并行 (ILP) 来隐藏 FMA 指令的延迟 (通常为 4-6 cycle)
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

// 核心 Kernel：测试 F32 __fmaf_rn 吞吐量
// 使用 intrinsic 函数强制生成 FFMA 指令
__global__ void fma_rn_stress_kernel(float* out, int n, float add_val, float mul_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = (float)tid * 0.0001f + (float)i * 0.1f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // F32 FMA Intrinsic (__fmaf_rn)
            // 语义: val = val * mul + add
            // 区别: 强制使用 Fused Multiply-Add，且采用 Round-to-Nearest 模式
            // 这对应汇编中的 FFMA R, R, R, R;
            val[i] = __fmaf_rn(val[i], mul_val, add_val); 
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
    // 需要大量线程来填满 SM 的 Warp Scheduler
    int blockSize = 256;
    // 根据 SM 数量动态调整 Block 数，确保占满 GPU
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数
    // Total Instructions = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 设定常数，保持数值在正常浮点范围内 (避免 NaN/Inf 影响某些特定架构的性能，虽然通常 FMA 并不受影响)
    float add_val = 0.00001f;
    float mul_val = 1.00000f; 

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    fma_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, add_val, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (F32 __fmaf_rn)..." << std::endl;
    
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
            fma_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, add_val, mul_val);
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
    double total_insts = ops_per_launch * total_launches; // 总指令数
    double ginst_per_sec = (total_insts / total_seconds) / 1e9; // G inst/s

    // 计算 TFLOPS 
    // 1个 FMA 指令 = 2 FLOPs (1次乘法 + 1次加法)
    double tflops = (ginst_per_sec * 2.0) / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Test Target: __fmaf_rn (Fused Multiply-Add Round-to-Nearest)" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Instructions: " << total_insts / 1e9 << " G inst" << std::endl;
    std::cout << "Instruction Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
