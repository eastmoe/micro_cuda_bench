#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含，用于 __half 和 __half2

// 宏定义：指令级并行度 (Instruction Level Parallelism)
// 使用 __half2 时，寄存器压力通常还好，8 路展开足以掩盖延迟
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

// --------------------------------------------------------------------------------
// 核心 Kernel：测试 FP16 Packed MUL 吞吐量
// 使用 __half2 类型，一条指令执行 2 次乘法
// --------------------------------------------------------------------------------
__global__ void mul_stress_kernel_fp16(__half2* out, int n, float mul_val_f) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 将传入的 float 参数转换为 __half2 (两个半精度数值相同)
    __half2 mul_val = __float2half2_rn(mul_val_f);

    // 初始化一组独立的寄存器变量
    __half2 val[ILP_FACTOR];
    
    // 初始化数据
    // 注意：FP16 精度较低，累乘容易溢出或下溢，这里主要测试指令吞吐
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float init_v = 1.0f + (float)tid * 0.0001f + (float)i * 0.001f;
        val[i] = __float2half2_rn(init_v);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，完全独立的乘法指令
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // 使用标准 FP16 乘法指令 (Packed)
            // 对应汇编通常为: HMUL2 R, R, R
            val[i] = __hmul2(val[i], mul_val); 
        }
    }

    // 简单的归约，防止被优化
    __half2 res = __float2half2_rn(0.0f);
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        res = __hadd2(res, val[i]);
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
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;

    // 检查架构是否支持 FP16
    if (props.major < 6) { // Pascal P100 (6.0) 之前 FP16 性能很差或不支持
        std::cout << "WARNING: Device architecture may not support fast native FP16 (Compute Capability < 6.0)." << std::endl;
        std::cout << "Results may reflect software emulation or reduced performance." << std::endl;
    }

    // 3. 配置 Grid 和 Block
    // 保持高 Occupancy
    int blockSize = 256;
    int numBlocks = props.multiProcessorCount * 128; // 增加 Block 数量确保填满 GPU
    int n = numBlocks * blockSize; // 总线程数（每个线程处理一个 __half2）

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads (processing __half2): " << n << std::endl;

    // 4. 分配显存
    // 这里的 n 是线程数，每个线程输出一个 __half2
    __half2* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__half2)));

    // 5. 计算指标
    // 每一个 loop 迭代包含 ILP_FACTOR 条指令
    // 每一条 __hmul2 指令包含 2 次浮点运算 (Packed)
    
    // 指令总数 (Giga Instructions)
    double inst_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // 浮点运算总数 (Giga FLOPS) -> 用于计算 TFLOPS
    double ops_per_launch = inst_per_launch * 2.0; 
    
    // 乘数设置
    // FP16 动态范围小，选取 1.0 附近的数。
    // 如果是纯粹为了测指令发射率，数值正确性不影响速度（除非遇到非规格化数，但在 1.0 附近不会）。
    // 这里传入 1.0001f，在 FP16 中可能被舍入，但只要不是 1.0 就能避免部分极端的编译器常量折叠（虽然 arg 传入已避免）。
    float mul_val = 1.0001f;

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    mul_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (FP16 Packed MUL - No Tensor Core)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        int batch_size = 50; // 稍微加大 batch 减少 CPU 开销
        for(int i=0; i<batch_size; i++) {
            mul_stress_kernel_fp16<<<numBlocks, blockSize>>>(d_out, n, mul_val);
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
    
    // 计算指令吞吐量 (GInst/s) - 对应 HMUL2 指令数
    double total_insts = inst_per_launch * total_launches;
    double ginst_per_sec = (total_insts / total_seconds) / 1e9;

    // 计算数学性能 (TFLOPS) - 对应实际乘法次数
    double total_math_ops = ops_per_launch * total_launches;
    double tflops = (total_math_ops / total_seconds) / 1e12;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "FP16 Instruction Throughput (HMUL2): " << ginst_per_sec << " G Inst/s (hmul2)" << std::endl;
    std::cout << "FP16 Arithmetic Performance:         " << tflops << " TFLOPS" << std::endl;
    //std::cout << "Note: Each HMUL2 instruction performs 2 Operations." << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
