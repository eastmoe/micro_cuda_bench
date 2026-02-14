#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度，用于隐藏指令延迟 (Instruction Level Parallelism)
// 标准倒数 (rcp.rn) 的延迟通常比近似倒数要高，需要足够的 ILP 来掩盖
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

// 核心 Kernel：__frcp_rn (IEEE 754 Round-to-Nearest Reciprocal) 测试
__global__ void frcp_rn_stress_kernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 初始化
    // 我们选择一个不会导致溢出或下溢的区间。
    // 逻辑：y = 1/x，再次 y = 1/y = x。数值会在两个值之间震荡。
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 偏移量 +1.5f 确保底数不为0，且数值量级适中
        val[i] = (float)tid * 0.00001f + (float)i * 0.1f + 1.5f; 
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 frcp_rn 之间没有依赖关系
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 核心修改点：使用 __frcp_rn()
            // 对应 PTX 指令: rcp.rn.f32
            // 这是一个符合 IEEE 754 标准的倒数操作，带有 Round-to-Nearest 舍入模式。
            // 它比 rcp.approx.ftz.f32 慢，但精度更高。
            val[i] = __frcp_rn(val[i]); 
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

    // 3. 配置 Grid 和 Block
    int blockSize = 256;
    // 确保有足够的 Block 填满 SM
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    frcp_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (__frcp_rn)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    // 循环发射 Kernel 直到时间耗尽
    while (true) {
        // 由于 __frcp_rn 可能比 rcp.approx 慢很多，我们适当减小 batch_size 以便更精准控制时间，
        // 但为了减少 CPU launch overhead 占比，这里设为 10 依然是合理的。
        int batch_size = 10; 
        for(int i=0; i<batch_size; i++) {
            frcp_rn_stress_kernel<<<numBlocks, blockSize>>>(d_out, n);
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
    std::cout << "FP32 __frcp_rn Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
