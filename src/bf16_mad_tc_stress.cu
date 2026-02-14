#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h> // 必须包含 WMMA 头文件

using namespace nvcuda;

// 宏定义：每个 Warp 内部持有的独立 Fragment 数量 (指令级并行度)
// 增加此值可以掩盖指令流水线延迟
#define ILP_FACTOR 16 

// 每次 Kernel 内部循环的次数
#define LOOP_ITER 1000

// WMMA 形状配置 (Ampere Standard for BF16)
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

// 核心 Kernel：测试 BF16 Tensor Core FMA 吞吐量
__global__ void wmma_stress_kernel_bf16(float* out, int total_warps) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (global_warp_id >= total_warps) return;

    // Fragment 定义
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[ILP_FACTOR];

    // 初始化
    wmma::fill_fragment(a_frag, __float2bfloat16(0.01f));
    wmma::fill_fragment(b_frag, __float2bfloat16(0.01f));
    #pragma unroll
    for (int i = 0; i < ILP_FACTOR; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    // 主循环：密集的 MMA 指令
    for (int k = 0; k < LOOP_ITER; k++) {
        #pragma unroll
        for (int i = 0; i < ILP_FACTOR; i++) {
            wmma::mma_sync(c_frag[i], a_frag, b_frag, c_frag[i]);
        }
    }

    // 防止编译器优化的强制依赖链
    #pragma unroll
    for (int i = 1; i < ILP_FACTOR; i++) {
        for (int t = 0; t < c_frag[0].num_elements; t++) {
             c_frag[0].x[t] += c_frag[i].x[t];
        }
    }

    // 防止死代码消除：写入 Global Memory
    int lane_id = threadIdx.x % 32;
    __shared__ float smem[256]; 
    float* smem_ptr = &smem[0];
    
    if (lane_id == 0) {
       wmma::store_matrix_sync(smem_ptr, c_frag[0], 16, wmma::mem_row_major);
       if (threadIdx.x == 0) {
           out[blockIdx.x] = smem[0]; 
       }
    }
}

int main(int argc, char* argv[]) {
    // 1. 参数解析
    double duration_target = 30.0; // 默认调整为 30秒，符合示例
    if (argc > 1) {
        try {
            duration_target = std::stod(argv[1]);
        } catch (...) {
            duration_target = 30.0;
        }
    }

    // 2. 获取设备信息
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    
    // 检查架构
    if (props.major < 8) {
        std::cerr << "Error: BF16 Tensor Core support requires Compute Capability 8.0 or higher." << std::endl;
        return 1;
    }

    // 3. 配置 Grid 和 Block
    int blockSize = 256; 
    int warpsPerBlock = blockSize / 32;
    // 保持足够的 SM 占用率，Ampere通常需要大量的Warp来隐藏延迟
    int numBlocks = props.multiProcessorCount * 16; 
    int totalWarps = numBlocks * warpsPerBlock;

    // 输出头部信息 (仿照示例格式)
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "SM Version: " << props.major << "." << props.minor << std::endl;
    std::cout << "Target Duration: " << duration_target << " seconds" << std::endl;
    std::cout << "Configuration: " << numBlocks << " blocks, " << blockSize << " threads/block." << std::endl;
    std::cout << "Total Warps: " << totalWarps << std::endl;

    // 4. 分配显存
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, numBlocks * sizeof(float)));

    // 5. 计算指标参数
    // FLOPs 计算: 2 * M * N * K
    double flops_per_mma = 2.0 * WMMA_M * WMMA_N * WMMA_K; 
    
    // 指令数计算: 每个 Loop 有 ILP_FACTOR 个 mma.sync
    double insts_per_warp_per_kernel = (double)LOOP_ITER * ILP_FACTOR;

    // 单次 Launch 的总工作量
    double ops_per_launch = (double)totalWarps * insts_per_warp_per_kernel * flops_per_mma;
    double insts_per_launch = (double)totalWarps * insts_per_warp_per_kernel;

    // 6. 预热
    std::cout << "Warming up..." << std::endl;
    wmma_stress_kernel_bf16<<<numBlocks, blockSize>>>(d_out, totalWarps);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 7. 开始测试
    std::cout << "Running stress test (Tensor Core FP16/BF16 mma.sync)..." << std::endl;
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    long long total_launches = 0;
    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventRecord(start));

    while (true) {
        int batch_size = 20; // 稍微减小 batch size 以便更精确控制时间
        for(int i=0; i<batch_size; i++) {
            wmma_stress_kernel_bf16<<<numBlocks, blockSize>>>(d_out, totalWarps);
        }
        total_launches += batch_size;

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        if (elapsed_ms >= duration_target * 1000.0) {
            break;
        }
    }

    // 8. 结果计算与输出
    double total_seconds = elapsed_ms / 1000.0;
    
    // TFLOPS
    double total_ops = ops_per_launch * total_launches;
    double tflops = (total_ops / total_seconds) / 1e12; 

    // G inst/s (Warp-level instructions)
    double total_insts = insts_per_launch * total_launches;
    double ginsts = (total_insts / total_seconds) / 1e9;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Elapsed Time: " << total_seconds << " s" << std::endl;
    std::cout << "Total Kernel Launches: " << total_launches << std::endl;
    std::cout << "Tensor Core Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Tensor Inst. Throughput: " << ginsts << " G inst/s (Warp-level mma.sync)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
