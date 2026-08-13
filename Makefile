# ==============================================================================
# Makefile for gfx906 MPI Multi-Precision Benchmark
# ==============================================================================

SHELL        := /bin/bash

# Compiler & Linker
CXX          := hipcc

# Target Executable Name
TARGET       := gfx906_mpi_multi_precision_bench

# Source Files
SRCS         := gfx906_mpi_multi_precision_bench.cpp

# Target GPU Architecture (Vega20)
OFFLOAD_ARCH := gfx906

# Optimization & C++ Standard
CXXFLAGS     := -O3 -std=c++11 --offload-arch=$(OFFLOAD_ARCH)

# Include Directories
MPI_INC      := /usr/include/x86_64-linux-gnu/mpi
INCLUDES     := -I$(MPI_INC)

# Linker Flags & Libraries
LDFLAGS      := 
LDLIBS       := -lmpi

# Combine default flags
ALL_CXXFLAGS := $(CXXFLAGS) $(INCLUDES)

# Multi-Node Cluster Execution Options
MPI_IFACE    := 192.168.178.0/24
MPI_NP       := 3
HOSTFILE     := cluster_hosts
RUN_SCRIPT   := run_node.sh

# ==============================================================================
# Build & Deploy Rules
# ==============================================================================

.PHONY: all deploy mi50 radeon-vii radeon-pro-vii mpirun run ssh-add clean

# Default target
all: mi50

# Master deploy target: builds remotes first, leaves local MI50 binary last
deploy: radeon-vii radeon-pro-vii mi50

# 1. MI50 Target (Local default host build)
mi50: $(SRCS)
	$(CXX) $(ALL_CXXFLAGS) $< -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# 2. Radeon VII Target: compiles with -DRadeon_vii and deploys
radeon-vii: $(SRCS)
	$(CXX) $(ALL_CXXFLAGS) -DRadeon_vii $< -o $(TARGET) $(LDFLAGS) $(LDLIBS)
	ssh Radeon-vii mkdir -p 1.0003-POPS
	scp $(TARGET) $(RUN_SCRIPT) Radeon-vii:1.0003-POPS

# 3. Radeon Pro VII Target: compiles with -DRadeon_pro_vii and deploys
radeon-pro-vii: $(SRCS)
	$(CXX) $(ALL_CXXFLAGS) -DRadeon_pro_vii $< -o $(TARGET) $(LDFLAGS) $(LDLIBS)
	ssh Radeon-pro-vii mkdir -p 1.0003-POPS
	scp $(TARGET) $(RUN_SCRIPT) Radeon-pro-vii:1.0003-POPS

# Run multi-node cluster benchmark across all nodes
mpirun:
	mpirun --mca btl_tcp_if_include $(MPI_IFACE) -np $(MPI_NP) --hostfile $(HOSTFILE) ./$(RUN_SCRIPT) ./$(TARGET)

# Alias for mpirun
run: mpirun

# Clean up local build artifacts
clean:
	rm -f $(TARGET) *.o
