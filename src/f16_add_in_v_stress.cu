#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含此头文件以支持 half 和 half2

// 宏定义：每个线程内部的展开程度
// ILP_FACTOR 越大，越能掩盖流水线延迟
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

// 核心 Kernel：测试 FP16 Vectorized ADD (Half2) 吞吐量
// 不使用 Tensor Core，而是使用 CUDA Core 的 FP16 SIMD 单元
__global__ void add_stress_kernel_fp16(float* out, int n, float add_val_f) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 将输入的 float 转换为 half2，两个槽位填入相同的值
    // half2 是一条指令处理两个 half 数据的关键
    half2 add_val = __float2half2_rn(add_val_f);

    // 初始化一组独立的寄存器变量 (half2 类型)
    half2 val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float init_val = (float)tid * 0.0001f + (float)i * 0.1f;
        val[i] = __float2half2_rn(init_val);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 add 之间没有依赖关系 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // FP16 向量化加法指令
            // __hadd2 对应汇编中的 HADD2 (或 HFMA2)，每个时钟周期执行 2 次加法
            val[i] = __hadd2(val[i], add_val); 
        }
    }

    // 归约结果并写回，防止被编译器视为死代码优化掉
    // 将 half2 拆解回 float 进行累加
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // __low2float 和 __high2float 分别提取 half2 中的低位和高位 half 并转为 float
        res += __low2float(val[i]) + __high2float(val[i]);
    }
    
    // 写入显存 (使用 float 数组作为输出容器方便验证，虽然这增加了最后写回的开销，但相比计算循环可忽略)
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
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;

    // 提示用户编译选项
    std::cout << "Note: Ensure this is compiled with -arch=sm_XX where XX >= 60 (e.g., -arch=sm_80)" << std::endl;

    // 3. 配置 Grid 和 Block
    // FP16 单元吞吐量通常是 FP32 的 2倍(对于 half2)，需要足够的并行度
    int blockSize = 256;
    int numBlocks = props.multiProcessorCount * 128; // 稍微增加 Grid 大小以确保填满
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    // 5. 计算每次 Kernel Launch 的总指令数 (Giga Instructions)
    // 这里的 Ops 指的是 "half2 指令数"
    double insts_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 定义加数
    float add_val = 1.00001f;

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    add_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, add_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 Vectorized ADD - half2)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // 每次发射一批 kernel，FP16 执行极快，增加 batch size
        int batch_size = 50; 
        for(int i=0; i<batch_size; i++) {
            add_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, add_val);
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
    
    // 总发射的指令数 (HADD2 指令)
    double total_insts = insts_per_launch * total_launches;
    
    // GInst/s: 每秒执行多少条 half2 指令
    double ginst_per_sec = (total_insts / total_seconds) / 1e9; 

    // TFLOPS 计算: 
    // 1 条 half2 指令 = 2 个浮点操作 (FLOPs)
    // 所以 TFLOPS = (GInst/s * 2) / 1000
    double tflops = (ginst_per_sec * 2.0) / 1000.0; 

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Total Instructions (half2): " << total_insts / 1e9 << " G Inst" << std::endl;
    std::cout << "Instruction Throughput: " << ginst_per_sec << " G Inst/s (half2)" << std::endl;
    std::cout << "Approx. FP16 Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
