# 1.0003-POPS
- show 1 peta operations INT4 performance with synthetic benchmark
- on 8× Instinct MI50, 1× Radeon vii and 1× Radeon pro vii GPUs
- on three workstations
- with OpenMPI
- and ```std::thread``` for multi-GPU workstation

Created with free gemini.google.com in many iterations.

## summary
```
=============================================================================
                   TOTAL 10-GPU CLUSTER PERFORMANCE                          
=============================================================================
 Combined INT4 Compute:  1000.25 TOPS
 Combined INT8 Compute:  500.16 TOPS
 Combined FP16 Compute:  259.68 TFLOPS
=============================================================================
```

Time for combined performance is measured after last GPU finished its synthetic benchmark.  
```sum_ops``` is sum of operations on all (10) GPUs.
```cpp
double run_mpi_benchmark_pass(int mpi_rank, int mpi_size, PrecisionMode mode, 
                              const std::string& label, uint64_t iterations, 
                              const std::string& hostname) {
    ....
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
    ...
    double aggregate_throughput = (sum_ops / wall_seconds) / 1e12;
    ...
    return aggregate_throughput;
}
```

```AMD Radeon VII``` is slightly faster than ```AMD Instinct MI50/MI60``` GPUs, and ```AMD Radeon (TM) Pro VII``` is slightly slower. Work is distributed slightly uneven in order to pass 1 POPS mark:
```cpp
int main(int argc, char** argv) {
...
#ifdef Radeon_vii
    const uint64_t iterations = 2047200ULL;
#elif Radeon_pro_vii
    const uint64_t iterations = 1976200ULL;
#else
    // Default for 7600x node
    uint64_t iterations = 2000000ULL;
#endif
...
}
```

## assembler verification of indeed 1 POPS INT4

128 INT4 operations per iteration in C++ ...
```cpp
__global__ void __launch_bounds__(256, 2) mi50_int4_kernel(uint64_t iterations, int* dummy_out) {
...
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
...
}
```

... can be seen in disassembly, so they are not compiled out:
```bash
hermann@Radeon-vii:~/1.0003-POPS$ roc-obj-ls ./gfx906_mpi_multi_precision_bench 
1       host-x86_64-unknown-linux--                                         file://./gfx906_mpi_multi_precision_bench#offset=24576&size=0
1       hipv4-amdgcn-amd-amdhsa--gfx906                                     file://./gfx906_mpi_multi_precision_bench#offset=24576&size=8640
hermann@Radeon-vii:~/1.0003-POPS$ roc-obj-extract "file://./gfx906_mpi_multi_precision_bench#offset=24576&size=8640"
hermann@Radeon-vii:~/1.0003-POPS$ ls -lst | head -2
total 60
12 -rw-rw-r-- 1 hermann hermann  8640 Aug 13 11:04 gfx906_mpi_multi_precision_bench-offset24576-size8640.co
hermann@Radeon-vii:~/1.0003-POPS$ /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx906 ./gfx906_mpi_multi_precision_bench-offset24576-size8640.co 

./gfx906_mpi_multi_precision_bench-offset24576-size8640.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001c00 <_Z16mi50_int4_kernelmPi>:
	s_load_dwordx4 s[0:3], s[4:5], 0x0                         // 000000001C00: C00A0002 00000000
...
...                             (16*8=128× v_dot8_i32_i4)
	v_mov_b32_e32 v9, 0                                        // 000000001C60: 7E120280
	v_dot8_i32_i4 v9, s4, v1, v9     <-----+                   // 000000001C64: D3AA4009 1C260204
...                                        |
        (110× v_dot8_i32_i4)               |
...                                        |
	v_dot8_i32_i4 v2, s4, v1, v2           |                   // 000000001FDC: D3AA4002 1C0A0204
	s_add_u32 s0, s0, -1                   |                   // 000000001FE4: 8000C100
	v_dot8_i32_i4 v9, s4, v1, v9           |                   // 000000001FE8: D3AA4009 1C260204
	v_dot8_i32_i4 v8, s4, v1, v8           |                   // 000000001FF0: D3AA4008 1C220204
	v_dot8_i32_i4 v7, s4, v1, v7           |                   // 000000001FF8: D3AA4007 1C1E0204
	v_dot8_i32_i4 v6, s4, v1, v6           |                   // 000000002000: D3AA4006 1C1A0204
	v_dot8_i32_i4 v5, s4, v1, v5           |                   // 000000002008: D3AA4005 1C160204
	v_dot8_i32_i4 v4, s4, v1, v4           |                   // 000000002010: D3AA4004 1C120204
	v_dot8_i32_i4 v3, s4, v1, v3           |                   // 000000002018: D3AA4003 1C0E0204
	v_dot8_i32_i4 v2, s4, v1, v2           |                   // 000000002020: D3AA4002 1C0A0204
	s_addc_u32 s1, s1, -1                  |                   // 000000002028: 8201C101
	v_dot8_i32_i4 v9, s4, v1, v9           |                   // 00000000202C: D3AA4009 1C260204
	v_dot8_i32_i4 v8, s4, v1, v8           |                   // 000000002034: D3AA4008 1C220204
	v_dot8_i32_i4 v7, s4, v1, v7           |                   // 00000000203C: D3AA4007 1C1E0204
	v_dot8_i32_i4 v6, s4, v1, v6           |                   // 000000002044: D3AA4006 1C1A0204
	v_dot8_i32_i4 v5, s4, v1, v5           |                   // 00000000204C: D3AA4005 1C160204
	v_dot8_i32_i4 v4, s4, v1, v4           |                   // 000000002054: D3AA4004 1C120204
	v_dot8_i32_i4 v3, s4, v1, v3           |                   // 00000000205C: D3AA4003 1C0E0204
	s_cmp_lg_u64 s[0:1], 0                 |                   // 000000002064: BF138000
	v_dot8_i32_i4 v2, s4, v1, v2           |                   // 000000002068: D3AA4002 1C0A0204  
	s_cbranch_scc1 65276             ------+                   // 000000002070: BF85FEFC <_Z16mi50_int4_kernelmPi+0x64>
...
hermann@Radeon-vii:~/1.0003-POPS$
```

## start ssh-agent
For password-less copying and mpirun: [start_agent.source](start_agent.source)
```bash
hermann@7600x:~$ cd 1.0003-POPS/
hermann@7600x:~/1.0003-POPS$ source start_agent.source 
Agent pid 3496
Enter passphrase for /home/hermann/.ssh/id_ed?????: 
Identity added: /home/hermann/.ssh/id_ed????? (hermann@stamm-wilbrandt.de)
hermann@7600x:~/1.0003-POPS$ 
```

## cluster 
- [8× Instinct MI50 (on 7600x)](https://media.printables.com/media/prints/1229329/rich_content/276cc0e3-cb78-4b6d-a263-fb68554d0c32/image.png)
- 1× GPU (on Radeon-vii)
- 1× GPU (on Radeon-pro-vii)

[cluster_hosts](cluster_hosts)  
```
# cluster_hosts
7600x slots=1
Radeon-vii slots=1
Radeon-pro-vii slots=1
```

## deploy
[deploy target](Makefile#L48-L49)
```bashhermann@7600x:~/1.0003-POPS$ make deploy
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi -DRadeon_vii gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
ssh Radeon-vii mkdir -p 1.0003-POPS
scp gfx906_mpi_multi_precision_bench run_node.sh Radeon-vii:1.0003-POPS
gfx906_mpi_multi_precision_bench                                                                100%   41KB  27.4MB/s   00:00    
run_node.sh                                                                                     100%   61   129.1KB/s   00:00    
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi -DRadeon_pro_vii gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
ssh Radeon-pro-vii mkdir -p 1.0003-POPS
scp gfx906_mpi_multi_precision_bench run_node.sh Radeon-pro-vii:1.0003-POPS
gfx906_mpi_multi_precision_bench                                                                100%   41KB  28.4MB/s   00:00    
run_node.sh                                                                                     100%   61   276.6KB/s   00:00    
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
hermann@7600x:~/1.0003-POPS$ 
```

## mpirun
[mpirun target](Makefile#L67-L69)
```bash
hermann@7600x:~/1.0003-POPS$ make mpirun
mpirun --mca btl_tcp_if_include 192.168.178.0/24 -np 3 --hostfile cluster_hosts ./run_node.sh ./gfx906_mpi_multi_precision_bench
=============================================================================
 RUNNING MULTI-NODE INT4 BENCHMARK PASS (2000000 iterations)
=============================================================================
Host            GPU ID    Device Name                   Time (ms)     Per-GPU TOPS    
--------------------------------------------------------------------------------------
7600x           0         AMD Instinct MI50/MI60        2517.80       99.95           
7600x           1         AMD Instinct MI50/MI60        2517.73       99.95           
7600x           2         AMD Instinct MI50/MI60        2517.78       99.95           
7600x           3         AMD Instinct MI50/MI60        2517.74       99.95           
7600x           4         AMD Instinct MI50/MI60        2518.55       99.92           
7600x           5         AMD Instinct MI50/MI60        2517.79       99.95           
7600x           6         AMD Instinct MI50/MI60        2517.81       99.95           
7600x           7         AMD Radeon Graphics           2516.64       100.00          
Radeon-vii      0         AMD Radeon VII                2517.88       102.31          
Radeon-pro-vii  0         AMD Radeon (TM) Pro VII       2517.34       98.78           
--------------------------------------------------------------------------------------
Active System GPUs:     10 / 10
Network Concurrent Wall:2518.894 ms
SYSTEM CLUSTER INT4:    1000.25 TOPS

=============================================================================
 RUNNING MULTI-NODE INT8 BENCHMARK PASS (2000000 iterations)
=============================================================================
Host            GPU ID    Device Name                   Time (ms)     Per-GPU TOPS    
--------------------------------------------------------------------------------------
7600x           0         AMD Instinct MI50/MI60        2516.98       49.99           
7600x           1         AMD Instinct MI50/MI60        2516.65       50.00           
7600x           2         AMD Instinct MI50/MI60        2516.64       50.00           
7600x           3         AMD Instinct MI50/MI60        2516.64       50.00           
7600x           4         AMD Instinct MI50/MI60        2518.13       49.97           
7600x           5         AMD Instinct MI50/MI60        2516.66       50.00           
7600x           6         AMD Instinct MI50/MI60        2516.63       50.00           
7600x           7         AMD Radeon Graphics           2516.66       50.00           
Radeon-vii      0         AMD Radeon VII                2518.51       51.14           
Radeon-pro-vii  0         AMD Radeon (TM) Pro VII       2516.78       49.40           
--------------------------------------------------------------------------------------
Active System GPUs:     10 / 10
Network Concurrent Wall:2518.744 ms
SYSTEM CLUSTER INT8:    500.16 TOPS

=============================================================================
 RUNNING MULTI-NODE FP16 BENCHMARK PASS (2000000 iterations)
=============================================================================
Host            GPU ID    Device Name                   Time (ms)     Per-GPU TFLOPS  
--------------------------------------------------------------------------------------
7600x           0         AMD Instinct MI50/MI60        2421.15       25.99           
7600x           1         AMD Instinct MI50/MI60        2420.97       25.99           
7600x           2         AMD Instinct MI50/MI60        2420.97       25.99           
7600x           3         AMD Instinct MI50/MI60        2420.97       25.99           
7600x           4         AMD Instinct MI50/MI60        2421.00       25.99           
7600x           5         AMD Instinct MI50/MI60        2420.98       25.99           
7600x           6         AMD Instinct MI50/MI60        2420.98       25.99           
7600x           7         AMD Radeon Graphics           2420.96       25.99           
Radeon-vii      0         AMD Radeon VII                2425.36       26.55           
Radeon-pro-vii  0         AMD Radeon (TM) Pro VII       2421.10       25.68           
--------------------------------------------------------------------------------------
Active System GPUs:     10 / 10
Network Concurrent Wall:2425.567 ms
SYSTEM CLUSTER FP16:    259.68 TFLOPS

=============================================================================
                   TOTAL 10-GPU CLUSTER PERFORMANCE                          
=============================================================================
 Combined INT4 Compute:  1000.25 TOPS
 Combined INT8 Compute:  500.16 TOPS
 Combined FP16 Compute:  259.68 TFLOPS
=============================================================================
hermann@7600x:~/1.0003-POPS$ 
```
