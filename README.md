# 1.0003-POPS
- show 1 peta operations INT4 performance with synthetic benchmark
- on 8× Instinct MI50, 1× Radeon vii and 1× Radeon pro vii GPUs
- on three workstations
- with OpenMPI

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

Time for combined performance is measured after last GPU finished its synthetic benchmark:
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
}
```

## start ssh-agent
```bash
hermann@7600x:~$ cd 1.0003-POPS/
hermann@7600x:~/1.0003-POPS$ source start_agent.source 
Agent pid 3496
Enter passphrase for /home/hermann/.ssh/id_ed?????: 
Identity added: /home/hermann/.ssh/id_ed????? (hermann@stamm-wilbrandt.de)
hermann@7600x:~/1.0003-POPS$ 
```

## cluster 
- 8× Instinct MI50 (on 7600x)
- 1× GPU (on Radeon-vii)
- 1× GPU (on Radeon-pro-vii)
```bash
pi@raspberrypi5:~/1.0003-POPS $ cat cluster_hosts 
# cluster_hosts
7600x slots=1
Radeon-vii slots=1
Radeon-pro-vii slots=1
pi@raspberrypi5:~/1.0003-POPS $ 
```

## deploy
```bash
hermann@7600x:~/pops$ make deploy
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi -DRadeon_vii gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
scp gfx906_mpi_multi_precision_bench run_node.sh Radeon-vii:1.0003-POPS
gfx906_mpi_multi_precision_bench                                                            100%   41KB  24.7MB/s   00:00    
run_node.sh                                                                                 100%   61    66.1KB/s   00:00    
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi -DRadeon_pro_vii gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
scp gfx906_mpi_multi_precision_bench run_node.sh Radeon-pro-vii:1.0003-POPS
gfx906_mpi_multi_precision_bench                                                            100%   41KB  30.2MB/s   00:00    
run_node.sh                                                                                 100%   61   369.0KB/s   00:00    
hipcc -O3 -std=c++11 --offload-arch=gfx906 -I/usr/include/x86_64-linux-gnu/mpi gfx906_mpi_multi_precision_bench.cpp -o gfx906_mpi_multi_precision_bench  -lmpi
hermann@7600x:~/1.0003-POPS$ 
```

## mpirun
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
