#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含此头文件以支持 __half 类型

// 宏定义：每个线程内部的展开程度
// 为了掩盖指令延迟，通常需要一定的指令级并行 (ILP)
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

// 核心 Kernel：测试 FP16 MUL (Scalar) 吞吐量
// 不使用 half2 (向量化)，不使用 Tensor Core
__global__ void mul_stress_kernel_fp16(__half* out, int n, __half mul_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 初始化一组独立的寄存器变量 (__half 类型)
    __half val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 使用 __float2half 进行类型转换
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 初始值稍作变化，避免全部相同
        float init_f = 1.0f + (float)tid * 0.0001f + (float)i * 0.01f;
        val[i] = __float2half(init_f);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 mul 之间没有依赖关系 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // FP16 乘法指令 (Scalar)
            // 在现代 GPU (sm_53+) 上，这将编译为 HFMA 或 HMUL 指令
            // 这里的 '*' 运算符由 cuda_fp16.h 重载
            val[i] = val[i] * mul_val; 
        }
    }

    // 归约结果并写回，防止被编译器视为死代码优化掉
    // 注意：FP16 累加容易溢出，但这里主要是为了测试乘法流水线
    __half res = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res = __hmul(res , val[i]);
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
    
    // 简单检查是否支持原生 FP16 (Compute Capability >= 5.3)
    if (props.major < 5 || (props.major == 5 && props.minor < 3)) {
        std::cerr << "Warning: Device compute capability " << props.major << "." << props.minor 
                  << " may not support native FP16 instructions. Performance may be low (software emulation)." << std::endl;
    }

    // 3. 配置 Grid 和 Block
    int blockSize = 256;
    // 保持足够的 Block 数以填满 SM
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    __half* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__half)));

    // 5. 计算每次 Kernel Launch 的总指令数
    // Total Ops = Threads * Loop_Iter * ILP_Factor
    double ops_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 定义乘数 (FP16)
    // FP16 精度较低，如果乘数离 1.0 太远，会迅速溢出(Inf)或归零(0)。
    // 1.0005 接近 1.0 + 2^-11
    __half mul_val_h = __float2half(1.0005f);

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    mul_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, mul_val_h);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 __hmul)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        int batch_size = 20; 
        for(int i=0; i<batch_size; i++) {
            mul_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, mul_val_h);
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
    double ginst_per_sec = (total_ops / total_seconds) / 1e9; // G inst/s

    // 计算 TFLOPS (FP16 Mul 算 1 FLOP)
    double tflops = ginst_per_sec / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Operations: " << total_ops / 1e9 << " G inst" << std::endl;
    std::cout << "FP16 Scalar MUL Throughput: " << ginst_per_sec << " G inst/s" << std::endl;
    std::cout << "Approx. MUL Performance: " << tflops << " TFLOPS (FP16)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    //std::cout << "Note: Without 'half2' (vectorization), throughput is typically \n"
    //          << "      lower than the theoretical peak FP16 performance of the GPU." << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
