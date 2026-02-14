#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
// 必须包含 BF16 头文件
#include <cuda_bf16.h>

// 宏定义：每个线程内部的展开程度 (Instruction Level Parallelism)
// 增加这个数值可以掩盖指令发射延迟
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

// 核心 Kernel：测试 BF16 MUL 吞吐量 (使用 Packed 模式)
// 不使用 Tensor Core，而是测试 CUDA Cores 的 BF16 处理能力
__global__ void mul_stress_kernel_bf16(__nv_bfloat162* out, int n, float mul_val_f) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 将传入的 float 转换为 packed BF16 (两个半宽浮点数都设为 mul_val_f)
    __nv_bfloat162 mul_val = __float2bfloat162_rn(mul_val_f);

    // 初始化一组独立的寄存器变量 (Packed 类型)
    __nv_bfloat162 val[ILP_FACTOR];
    
    // 给每个变量赋不同的初值
    // 注意：对于乘法，不能初始化为 0，否则结果永远是 0。
    // 这里初始化为 1.0 左右的数值。
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        float init_v = 1.0f + (float)tid * 0.00001f + (float)i * 0.001f;
        val[i] = __float2bfloat162_rn(init_v);
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        // 展开计算，所有的 mul 之间没有依赖关系 (ILP)
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // BF16 Packed 乘法指令
            // 对应汇编通常是 HMUL2 (BF16 variant)
            // 这是一个非 Tensor Core 的 SIMT 指令，每条指令执行 2 次乘法
            val[i] = val[i] * mul_val; 
        }
    }

    // 归约结果并写回，防止被编译器视为死代码优化掉
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        // 将 bfloat162 转回两个 float 并累加
        float2 tmp = __bfloat1622float2(val[i]);
        res += (tmp.x + tmp.y);
    }
    
    // 写入显存产生的副作用，确保计算逻辑被保留
    out[tid] = __float2bfloat162_rn(res);
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
    
    // 检查架构是否支持 BF16 (Ampere sm_80 +)
    // 注意：虽然 Turing (sm_75) 有 TensorCore BF16 实验性支持，
    // 但标准 CUDA Core BF16 指令主要在 Ampere 及之后普及。
    if (props.major < 8) {
        std::cerr << "Error: Native BF16 support requires Compute Capability 8.0 or higher." << std::endl;
        return 1;
    }

    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;

    // 3. 配置 Grid 和 Block
    int blockSize = 256;
    // 保持足够的 Wavefronts 以占满 GPU
    int numBlocks = props.multiProcessorCount * 64; 
    int n = numBlocks * blockSize;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Threads: " << n << std::endl;

    // 4. 分配显存
    __nv_bfloat162* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__nv_bfloat162)));

    // 5. 计算指标
    // 每次 Kernel 的“指令”数 (Instructions)
    // 1 个 Loop = ILP_FACTOR 条 Packed MUL 指令
    double insts_per_launch = (double)n * LOOP_ITER * ILP_FACTOR;
    
    // Total Ops (FLOPs) = 指令数 * 2 (因为是 Packed 2-way SIMD)
    double ops_per_launch = insts_per_launch * 2.0; 

    // 定义乘数：选一个非常接近 1.0 的数，避免 overflow 或 underflow
    // 这样纯粹为了测试 pipeline 吞吐
    float mul_val = 1.000001f;

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    mul_stress_kernel_bf16<<<numBlocks, blockSize>>>(d_out, n, mul_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (BF16 Packed MUL without TensorCore)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        // 批量发射 Kernel，减少 CPU Launch Overhead
        int batch_size = 20; 
        for(int i=0; i<batch_size; i++) {
            mul_stress_kernel_bf16<<<numBlocks, blockSize>>>(d_out, n, mul_val);
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
    
    // 总指令数 (Giga Instructions)
    double total_insts = insts_per_launch * total_launches;
    // 总浮点运算数 (Tera FLOPS)
    double total_ops = ops_per_launch * total_launches;

    double ginst_per_sec = (total_insts / total_seconds) / 1e9; // G Inst/s
    double tflops = (total_ops / total_seconds) / 1e12;         // T FLOPS

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Instruction Throughput: " << ginst_per_sec << " G inst/s (Packed BF16)" << std::endl;
    std::cout << "BF16 Performance: " << tflops << " TFLOPS" << std::endl;
    //std::cout << "Note: BF16 TFLOPS = 2 * Instruction Throughput (due to packed math)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
