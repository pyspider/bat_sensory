# WE-FDTD Build Helper Makefile
# Build helpers for compiling WE-FDTD with nvcc and simple checks

# Configuration
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
MPI_PATH ?= /usr/lib/x86_64-linux-gnu/openmpi
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11 -Xcompiler -fPIC
CUDA_ARCH = -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_70,code=sm_70

# Include paths
INCLUDES = -I$(CUDA_PATH)/include
MPI_INCLUDES = -I$(MPI_PATH)/include

# Library paths
LIBRARIES = -L$(CUDA_PATH)/lib64 -lcudart -lcufft
MPI_LIBRARIES = -L$(MPI_PATH)/lib -lmpi

# Source and target directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Default target
.PHONY: all clean help check-deps test

all: check-deps $(BIN_DIR)/we-fdtd

help:
	@echo "WE-FDTD Build Helper"
	@echo "===================="
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build WE-FDTD executable"
	@echo "  clean       - Clean build artifacts"
	@echo "  check-deps  - Check build dependencies"
	@echo "  test        - Run basic functionality tests"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH   - Path to CUDA installation ($(CUDA_PATH))"
	@echo "  MPI_PATH    - Path to MPI installation ($(MPI_PATH))"
	@echo ""
	@echo "Example usage:"
	@echo "  make all"
	@echo "  make CUDA_PATH=/opt/cuda"
	@echo "  make clean && make"

# Check dependencies
check-deps:
	@echo "Checking build dependencies..."
	@echo -n "CUDA compiler (nvcc): "
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "✓ Found at $(NVCC)"; \
		$(NVCC) --version | grep "release" | head -1; \
	else \
		echo "✗ Not found"; \
		echo "Please install CUDA toolkit or set CUDA_PATH"; \
		exit 1; \
	fi
	@echo -n "C++ compiler (g++): "
	@if command -v $(CXX) >/dev/null 2>&1; then \
		echo "✓ Found"; \
		$(CXX) --version | head -1; \
	else \
		echo "✗ Not found"; \
		echo "Please install build-essential"; \
		exit 1; \
	fi
	@echo -n "MPI support: "
	@if [ -d "$(MPI_PATH)" ]; then \
		echo "✓ Found at $(MPI_PATH)"; \
	else \
		echo "⚠ Not found (optional)"; \
		echo "Install libopenmpi-dev for MPI support"; \
	fi
	@echo -n "GPU devices: "
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi -L | wc -l | tr -d '\n'; \
		echo " GPU(s) detected"; \
	else \
		echo "⚠ nvidia-smi not found"; \
	fi

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Mock WE-FDTD sources (since we don't have the actual source)
$(SRC_DIR):
	@echo "Creating mock WE-FDTD source structure..."
	mkdir -p $(SRC_DIR)
	@echo '// Mock WE-FDTD main.cpp' > $(SRC_DIR)/main.cpp
	@echo '#include <iostream>' >> $(SRC_DIR)/main.cpp
	@echo '#include <cuda_runtime.h>' >> $(SRC_DIR)/main.cpp
	@echo 'int main(int argc, char* argv[]) {' >> $(SRC_DIR)/main.cpp
	@echo '    std::cout << "WE-FDTD Mock Simulator v1.0" << std::endl;' >> $(SRC_DIR)/main.cpp
	@echo '    int deviceCount = 0;' >> $(SRC_DIR)/main.cpp
	@echo '    cudaGetDeviceCount(&deviceCount);' >> $(SRC_DIR)/main.cpp
	@echo '    std::cout << "CUDA devices available: " << deviceCount << std::endl;' >> $(SRC_DIR)/main.cpp
	@echo '    if (argc > 1) {' >> $(SRC_DIR)/main.cpp
	@echo '        std::cout << "Processing arguments..." << std::endl;' >> $(SRC_DIR)/main.cpp
	@echo '        for (int i = 1; i < argc; i++) {' >> $(SRC_DIR)/main.cpp
	@echo '            std::cout << "  arg[" << i << "]: " << argv[i] << std::endl;' >> $(SRC_DIR)/main.cpp
	@echo '        }' >> $(SRC_DIR)/main.cpp
	@echo '    }' >> $(SRC_DIR)/main.cpp
	@echo '    std::cout << "Simulation completed successfully." << std::endl;' >> $(SRC_DIR)/main.cpp
	@echo '    return 0;' >> $(SRC_DIR)/main.cpp
	@echo '}' >> $(SRC_DIR)/main.cpp

# Build WE-FDTD executable
$(BIN_DIR)/we-fdtd: $(SRC_DIR) | $(BUILD_DIR) $(BIN_DIR)
	@echo "Building WE-FDTD..."
	@echo "Note: This is a mock build for integration testing"
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) \
		-o $@ $(SRC_DIR)/main.cpp $(LIBRARIES)
	@echo "Build completed: $@"

# Test the built executable
test: $(BIN_DIR)/we-fdtd
	@echo "Running basic functionality tests..."
	@echo "Test 1: Check executable runs"
	./$(BIN_DIR)/we-fdtd --help || true
	@echo ""
	@echo "Test 2: Check CUDA detection"
	./$(BIN_DIR)/we-fdtd
	@echo ""
	@echo "Test 3: Check argument processing"
	./$(BIN_DIR)/we-fdtd -i test.dat -o output/
	@echo ""
	@echo "Basic tests completed ✓"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(SRC_DIR)
	@echo "Clean completed"

# Install (copy to system path)
install: $(BIN_DIR)/we-fdtd
	@echo "Installing WE-FDTD to /usr/local/bin..."
	sudo cp $(BIN_DIR)/we-fdtd /usr/local/bin/
	@echo "Installation completed"

# Uninstall
uninstall:
	@echo "Removing WE-FDTD from /usr/local/bin..."
	sudo rm -f /usr/local/bin/we-fdtd
	@echo "Uninstallation completed"

# Development helpers
dev-setup:
	@echo "Setting up development environment..."
	@echo "This would typically:"
	@echo "  - Install CUDA toolkit"
	@echo "  - Install MPI libraries"
	@echo "  - Set up IDE configurations"
	@echo "  - Download sample datasets"
	@echo ""
	@echo "For actual setup, please refer to docs/FDTD_INTEGRATION.md"

# GPU info
gpu-info:
	@echo "GPU Information:"
	@echo "==============="
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi; \
	else \
		echo "nvidia-smi not available"; \
	fi

.PHONY: install uninstall dev-setup gpu-info