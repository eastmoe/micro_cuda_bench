#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

// 宏定义：每个线程内部的展开程度
// __fdividef 通常比标准 div 快很多（通常编译为 rcp + mul），
// 但为了保持与标准 div 测试的可比性，我们维持相同的 ILP_FACTOR。
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

// 核心 Kernel：测试 __fdividef 吞吐量
// __fdividef 是 CUDA 的内置函数，执行快速除法（通常是倒数近似值乘法）
__global__ void fdividef_stress_kernel(float* out, int n, float div_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量
    float val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        val[i] = (float)tid * 0.0001f + (float)i * 0.1f + 1.0f;
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，利用指令级并行 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用 __fdividef 内置函数
            // 这通常比标准的 operator/ 快，因为它跳过了一些 IEEE 754 的边缘情况处理（如非正规化数）
            val[i] = __fdividef(val[i], div_val); 
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
    int blockSize = 256;
    // 保持足够的线程数以占满 GPU
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数
    // Total Ops = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 定义除数
    // 使用接近 1.0 的数
    float div_val = 1.00001f; 

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    fdividef_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, div_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (F32 __fdividef)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // __fdividef 吞吐量很高，为了减少 Kernel Launch 开销占比，
        // 可以适当增加 batch_size，或者保持原样。这里稍微增加一点以应对更高的吞吐。
        int batch_size = 20; 
        for(int i=0; i<batch_size; i++) {
            fdividef_stress_kernel<<<numBlocks, blockSize>>>(d_out, n, div_val);
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
    double ginst_per_sec = (total_ops / total_seconds) / 1e9; // G inst/s

    // 计算 TFLOPS (此处特指 __fdividef Operations)
    double tflops = ginst_per_sec / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Test: __fdividef (Intrinsic Fast Division)" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "FP32 __fdividef Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "Approx. Performance: " << tflops << " TFLOPS (__fdividef Ops)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
