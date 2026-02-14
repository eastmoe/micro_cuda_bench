#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // 必须包含，用于 Tensor Core (WMMA) 支持

using namespace nvcuda;

// 宏定义：Warp 内部的展开程度 (Instruction Level Parallelism)
#define ILP_FACTOR 16

// 每次 Kernel 内部循环的次数
#define LOOP_ITER 1000

// WMMA 矩阵维度设定 (m-n-k)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 检查 CUDA 错误的辅助函数
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// 核心 Kernel：测试 Tensor Core FP16 HMMA 吞吐量
// 修改 1: 输出指针类型改为 __half* 以匹配 accumulator 类型
__global__ void tensor_core_stress_kernel(__half* out, int n) {
    // 计算全局 Warp ID
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int totalWarps = n / 32;
    if (warpId >= totalWarps) return;

    // 声明 WMMA 片段 (Fragments)
    // Accumulator 使用 half 类型 (纯 FP16 模式)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[ILP_FACTOR];

    // 初始化片段
    wmma::fill_fragment(a_frag, __float2half(0.01f));
    wmma::fill_fragment(b_frag, __float2half(0.01f));

    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

    // 主循环
    for (int k = 0; k < LOOP_ITER; k++) {
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            // Tensor Core 矩阵乘加运算
            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
    }

        // ================== 修复开始 ==================
    // 强制依赖：将 c_frag[1...15] 全部累加到 c_frag[0]
    // 只有这样，编译器才不敢删除前面的计算指令
    #pragma unroll
    for (int i = 1; i < ILP_FACTOR; i++) {
        for (int t = 0; t < c_frag[0].num_elements; t++) {
            // 注意：这里使用 FP16 加法
            // 虽然访问 fragment 内部 .x 是为了 hack，但在测试代码中是有效的
            c_frag[0].x[t] = __hadd(c_frag[0].x[t], c_frag[i].x[t]);
        }
    }
    // ================== 修复结束 ==================

    // 防止编译器优化掉计算过程
    // 将 Warp 中第一个 accumulator 的数据写回
    // 修改 2: out 是 __half*，c_frag 是 half 类型，现在类型匹配了
    int laneId = threadIdx.x % 32;
    if (laneId == 0) {
         wmma::store_matrix_sync(out + warpId * WMMA_M * WMMA_N, c_frag[0], WMMA_N, wmma::mem_row_major);
    }
}

int main(int argc, char* argv[]) {
    // 1. 参数解析
    double duration_target = 5.0; 
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
    std::cout << "SM Version: " << props.major << "." << props.minor << std::endl;
    
    if (props.major < 7) {
        std::cerr << "Error: Tensor Cores require Volta (SM 7.0) or later." << std::endl;
        return 1;
    }

    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;

    // 3. 配置 Grid 和 Block
    int blockSize = 256; 
    int numBlocks = props.multiProcessorCount * 32; 
    int totalThreads = numBlocks * blockSize;
    int totalWarps = totalThreads / 32;

    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Warps: " << totalWarps << std::endl;

    // 4. 分配显存
    // 修改 3: 使用 __half* 类型，并按 sizeof(__half) 分配内存
    __half* d_out;
    size_t memSize = totalWarps * WMMA_M * WMMA_N * sizeof(__half);
    CHECK_CUDA(cudaMalloc(&d_out, memSize));

    // 5. 计算 FLOPs 统计
    // 1个 mma.sync (16x16x16) = 2 * 16 * 16 * 16 = 8192 FLOPs
    double ops_per_mma = 2.0 * WMMA_M * WMMA_N * WMMA_K; 
    double ops_per_launch = (double)totalWarps * LOOP_ITER * ILP_FACTOR * ops_per_mma;

    // 6. 预热 (Warmup)
    std::cout << "Warming up..." << std::endl;
    tensor_core_stress_kernel<<<numBlocks, blockSize>>>(d_out, totalThreads);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (Tensor Core FP16 mma.sync)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        int batch_size = 50; 
        for(int i=0; i<batch_size; i++) {
            tensor_core_stress_kernel<<<numBlocks, blockSize>>>(d_out, totalThreads);
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
    double total_flops = ops_per_launch * total_launches;
    double tflops = (total_flops / total_seconds) / 1e12; 

    double total_tensor_insts = (double)totalWarps * LOOP_ITER * ILP_FACTOR * total_launches;
    double g_tensor_inst_sec = (total_tensor_insts / total_seconds) / 1e9;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Tensor Core Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Tensor Inst. Throughput: " << g_tensor_inst_sec << " G inst/s (Warp-level mma.sync)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 清理
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
