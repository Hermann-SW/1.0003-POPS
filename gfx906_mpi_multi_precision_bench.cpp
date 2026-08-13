#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <string>
#include <unistd.h>

enum PrecisionMode {
    PREC_INT4,
    PREC_INT8,
    PREC_FP16
};

// ---------------------------------------------------------------------------
// Compute Kernels
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256, 2) mi50_int4_kernel(uint64_t iterations, int* dummy_out) {
    int src0 = 0x12345678;
    int src1 = 0x87654321;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    #pragma unroll 1
    for (uint64_t i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc0 = __builtin_amdgcn_sdot8(src0, src1, acc0, false);
            acc1 = __builtin_amdgcn_sdot8(src0, src1, acc1, false);
            acc2 = __builtin_amdgcn_sdot8(src0, src1, acc2, false);
            acc3 = __builtin_amdgcn_sdot8(src0, src1, acc3, false);
            acc4 = __builtin_amdgcn_sdot8(src0, src1, acc4, false);
            acc5 = __builtin_amdgcn_sdot8(src0, src1, acc5, false);
            acc6 = __builtin_amdgcn_sdot8(src0, src1, acc6, false);
            acc7 = __builtin_amdgcn_sdot8(src0, src1, acc7, false);
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dummy_out = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    }
}

__global__ void __launch_bounds__(256, 2) mi50_int8_kernel(uint64_t iterations, int* dummy_out) {
    int src0 = 0x12345678;
    int src1 = 0x87654321;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    #pragma unroll 1
    for (uint64_t i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc0 = __builtin_amdgcn_sdot4(src0, src1, acc0, false);
            acc1 = __builtin_amdgcn_sdot4(src0, src1, acc1, false);
            acc2 = __builtin_amdgcn_sdot4(src0, src1, acc2, false);
            acc3 = __builtin_amdgcn_sdot4(src0, src1, acc3, false);
            acc4 = __builtin_amdgcn_sdot4(src0, src1, acc4, false);
            acc5 = __builtin_amdgcn_sdot4(src0, src1, acc5, false);
            acc6 = __builtin_amdgcn_sdot4(src0, src1, acc6, false);
            acc7 = __builtin_amdgcn_sdot4(src0, src1, acc7, false);
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dummy_out = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    }
}

__global__ void __launch_bounds__(256, 2) mi50_fp16_kernel(uint64_t iterations, half2* dummy_out) {
    half2 src0 = __float2half2_rn(1.001f);
    half2 src1 = __float2half2_rn(0.999f);
    half2 acc0 = __float2half2_rn(0.1f);
    half2 acc1 = __float2half2_rn(0.2f);
    half2 acc2 = __float2half2_rn(0.3f);
    half2 acc3 = __float2half2_rn(0.4f);
    half2 acc4 = __float2half2_rn(0.5f);
    half2 acc5 = __float2half2_rn(0.6f);
    half2 acc6 = __float2half2_rn(0.7f);
    half2 acc7 = __float2half2_rn(0.8f);

    #pragma unroll 1
    for (uint64_t i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc0 = __hfma2(src0, src1, acc0);
            acc1 = __hfma2(src0, src1, acc1);
            acc2 = __hfma2(src0, src1, acc2);
            acc3 = __hfma2(src0, src1, acc3);
            acc4 = __hfma2(src0, src1, acc4);
            acc5 = __hfma2(src0, src1, acc5);
            acc6 = __hfma2(src0, src1, acc6);
            acc7 = __hfma2(src0, src1, acc7);
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dummy_out = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    }
}

// ---------------------------------------------------------------------------
// Structures & Worker Thread
// ---------------------------------------------------------------------------

struct GpuResultData {
    char hostname[64];
    int local_dev_id;
    char dev_name[64];
    float kernel_ms;
    double tops;
    double total_ops;
    int success;
};

void worker_thread(int dev_id, PrecisionMode mode, uint64_t iterations, GpuResultData* result,
                   const std::string& hostname, std::atomic<int>& ready_count, 
                   std::atomic<bool>& start_flag) {
    snprintf(result->hostname, sizeof(result->hostname), "%s", hostname.c_str());
    result->local_dev_id = dev_id;
    result->success = 0;

    if (hipSetDevice(dev_id) != hipSuccess) return;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev_id) != hipSuccess) return;
    snprintf(result->dev_name, sizeof(result->dev_name), "%s", prop.name);

    // Dynamically query Compute Units (typically 60 for Vega 20)
    const int num_blocks = prop.multiProcessorCount * 4;
    const int threads_per_block = 256;
    const double total_threads = (double)num_blocks * threads_per_block;

    double ops_per_thread_per_iter = 0.0;
    if (mode == PREC_INT4)      ops_per_thread_per_iter = 128.0 * 16.0; // 2048 ops
    else if (mode == PREC_INT8) ops_per_thread_per_iter = 128.0 * 8.0;  // 1024 ops
    else if (mode == PREC_FP16) ops_per_thread_per_iter = 128.0 * 4.0;  // 512 FLOPs

    result->total_ops = ops_per_thread_per_iter * (double)iterations * total_threads;

    void* d_out = nullptr;
    if (hipMalloc(&d_out, sizeof(double)) != hipSuccess) return;

    // Local warmup
    if (mode == PREC_INT4)      mi50_int4_kernel<<<num_blocks, threads_per_block>>>(10000ULL, (int*)d_out);
    else if (mode == PREC_INT8) mi50_int8_kernel<<<num_blocks, threads_per_block>>>(10000ULL, (int*)d_out);
    else if (mode == PREC_FP16) mi50_fp16_kernel<<<num_blocks, threads_per_block>>>(10000ULL, (half2*)d_out);

    if (hipDeviceSynchronize() != hipSuccess) {
        hipFree(d_out);
        return;
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    ready_count++;

    while (!start_flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // Synchronized execution
    hipEventRecord(start);
    if (mode == PREC_INT4)      mi50_int4_kernel<<<num_blocks, threads_per_block>>>(iterations, (int*)d_out);
    else if (mode == PREC_INT8) mi50_int8_kernel<<<num_blocks, threads_per_block>>>(iterations, (int*)d_out);
    else if (mode == PREC_FP16) mi50_fp16_kernel<<<num_blocks, threads_per_block>>>(iterations, (half2*)d_out);
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    double seconds = milliseconds / 1000.0;
    result->kernel_ms = milliseconds;
    result->tops = (result->total_ops / seconds) / 1e12;
    result->success = 1;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_out);
}

// ---------------------------------------------------------------------------
// Multi-Node Benchmark Execution Pass
// ---------------------------------------------------------------------------

double run_mpi_benchmark_pass(int mpi_rank, int mpi_size, PrecisionMode mode, 
                              const std::string& label, uint64_t iterations, 
                              const std::string& hostname) {
    int local_dev_count = 0;
    hipGetDeviceCount(&local_dev_count);

    if (mpi_rank == 0) {
        std::cout << "=============================================================================\n";
        std::cout << " RUNNING MULTI-NODE " << label << " BENCHMARK PASS (" << iterations << " iterations)\n";
        std::cout << "=============================================================================\n";
    }

    std::vector<GpuResultData> local_results(local_dev_count);
    std::vector<std::thread> threads;
    std::atomic<int> ready_count(0);
    std::atomic<bool> start_flag(false);

    for (int i = 0; i < local_dev_count; ++i) {
        threads.emplace_back(worker_thread, i, mode, iterations, &local_results[i], 
                             hostname, std::ref(ready_count), std::ref(start_flag));
    }

    // Wait until all local GPU threads are warmed up
    while (ready_count.load(std::memory_order_relaxed) < local_dev_count) {
        std::this_thread::yield();
    }

    // Global Network Barrier across all physical nodes
    MPI_Barrier(MPI_COMM_WORLD);

    double wall_start = MPI_Wtime();
    start_flag.store(true, std::memory_order_release);

    for (auto& t : threads) {
        t.join();
    }

    // Wait for all nodes to finish calculation before measuring final wall time
    MPI_Barrier(MPI_COMM_WORLD);
    double wall_stop = MPI_Wtime();
    double wall_seconds = wall_stop - wall_start;

    // Gather result counts from all nodes
    std::vector<int> recv_counts(mpi_size, 0);
    MPI_Gather(&local_dev_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_gpus_system = 0;
    std::vector<int> displs(mpi_size, 0);
    if (mpi_rank == 0) {
        for (int i = 0; i < mpi_size; ++i) {
            displs[i] = total_gpus_system;
            total_gpus_system += recv_counts[i];
        }
    }

    std::vector<GpuResultData> global_results(total_gpus_system);

    // Convert counts to byte sizes for MPI_Gatherv
    std::vector<int> recv_bytes(mpi_size, 0);
    std::vector<int> byte_displs(mpi_size, 0);
    if (mpi_rank == 0) {
        for (int i = 0; i < mpi_size; ++i) {
            recv_bytes[i] = recv_counts[i] * sizeof(GpuResultData);
            byte_displs[i] = displs[i] * sizeof(GpuResultData);
        }
    }

    MPI_Gatherv(local_results.data(), local_dev_count * sizeof(GpuResultData), MPI_BYTE,
                global_results.data(), recv_bytes.data(), byte_displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    double aggregate_throughput = 0.0;

    if (mpi_rank == 0) {
        std::cout << std::left << std::setw(16) << "Host"
                  << std::setw(10) << "GPU ID"
                  << std::setw(30) << "Device Name"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(16) << (mode == PREC_FP16 ? "Per-GPU TFLOPS" : "Per-GPU TOPS") << "\n";
        std::cout << std::string(86, '-') << "\n";

        double total_system_ops = 0.0;
        int active_gpus = 0;

        for (const auto& r : global_results) {
            if (r.success) {
                std::cout << std::left << std::setw(16) << r.hostname
                          << std::setw(10) << r.local_dev_id
                          << std::setw(30) << r.dev_name
                          << std::setw(14) << std::fixed << std::setprecision(2) << r.kernel_ms
                          << std::setw(16) << std::setprecision(2) << r.tops << "\n";
                total_system_ops += r.total_ops;
                active_gpus++;
            }
        }

        aggregate_throughput = (total_system_ops / wall_seconds) / 1e12;

        std::cout << std::string(86, '-') << "\n";
        std::cout << "Active System GPUs:     " << active_gpus << " / " << total_gpus_system << "\n";
        std::cout << "Network Concurrent Wall:" << std::fixed << std::setprecision(3) << wall_seconds * 1000.0 << " ms\n";
        std::cout << "SYSTEM CLUSTER " << label << ":    " << std::setprecision(2) << aggregate_throughput 
                  << (mode == PREC_FP16 ? " TFLOPS" : " TOPS") << "\n\n";
    }

    return aggregate_throughput;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank = 0, mpi_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    char host_buf[64];
    gethostname(host_buf, sizeof(host_buf));
    std::string hostname(host_buf);

    int local_dev_count = 0;
    hipGetDeviceCount(&local_dev_count);

#ifdef Radeon_vii
    const uint64_t iterations = 2047200ULL;
#elif Radeon_pro_vii
    const uint64_t iterations = 1976200ULL;
#else
    // Default for 7600x node
    uint64_t iterations = 2000000ULL;
#endif

    double agg_int4 = run_mpi_benchmark_pass(mpi_rank, mpi_size, PREC_INT4, "INT4", iterations, hostname);
    double agg_int8 = run_mpi_benchmark_pass(mpi_rank, mpi_size, PREC_INT8, "INT8", iterations, hostname);
    double agg_fp16 = run_mpi_benchmark_pass(mpi_rank, mpi_size, PREC_FP16, "FP16", iterations, hostname);

    if (mpi_rank == 0) {
        std::cout << "=============================================================================\n";
        std::cout << "                   TOTAL 10-GPU CLUSTER PERFORMANCE                          \n";
        std::cout << "=============================================================================\n";
        std::cout << " Combined INT4 Compute:  " << std::fixed << std::setprecision(2) << agg_int4 << " TOPS\n";
        std::cout << " Combined INT8 Compute:  " << std::fixed << std::setprecision(2) << agg_int8 << " TOPS\n";
        std::cout << " Combined FP16 Compute:  " << std::fixed << std::setprecision(2) << agg_fp16 << " TFLOPS\n";
        std::cout << "=============================================================================\n";
    }

    MPI_Finalize();
    return 0;
}
