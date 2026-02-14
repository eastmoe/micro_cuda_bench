#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：指令级并行度 (ILP)
// 通过展开循环，让一个线程由于指令依赖较少，能填满流水线
// 对于 __fsqrt_rn 这种延迟较高的指令，较高的 ILP 有助于测出峰值吞吐
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

// 核心 Kernel：测试 __fsqrt_rn (Round-to-Nearest Square Root)
__global__ void fsqrt_rn_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 寄存器数组，用于指令展开
    float val[ILP_FACTOR];
    
    // 初始化：确保是正数，避免 NaN
    // 使用较大的数开始，因为 sqrt 会让数字迅速变小趋向于 1.0
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = fabsf((float)tid) + 1000.0f + (float)i * 10.0f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // --- 核心修改 ---
            // __fsqrt_rn: 强制生成 sqrt.rn.f32 指令
            // 无论编译选项是否开启 fast-math，该指令都会执行 IEEE 754 标准精度的平方根
            val[i] = __fsqrt_rn(val[i]); 
        }
    }

    // 归约结果并写回，防止被编译器优化掉 (Dead Code Elimination)
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
    std::cout << "Test Instruction: F32 __fsqrt_rn (sqrt.rn.f32)" << std::endl;

    // 3. 配置 Grid 和 Block
    // 保持高占用率
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
    fsqrt_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
        // 批量发射以减少 CPU 开销
        int batch_size = 20; 
        for(int i=0; i<batch_size; i++) {
            fsqrt_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    double total_ops = ops_per_launch * total_launches;
    double ginst_per_sec = (total_ops / total_seconds) / 1e9;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "F32 __fsqrt_rn Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
