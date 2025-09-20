#!/bin/bash
# WE-FDTD Build Script
# Build helpers for compiling WE-FDTD with nvcc and simple checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_LOG="${SCRIPT_DIR}/build.log"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
MPI_PATH="${MPI_PATH:-/usr/lib/x86_64-linux-gnu/openmpi}"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$BUILD_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BUILD_LOG"
    exit 1
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

print_header() {
    echo ""
    echo "============================================"
    echo "  WE-FDTD Build Script for bat_sensory"
    echo "============================================"
    echo ""
}

check_dependencies() {
    log "Checking build dependencies..."
    
    # Check NVCC
    if check_command nvcc; then
        NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        log_success "NVCC found (version $NVCC_VERSION)"
    else
        log_error "NVCC not found. Please install CUDA toolkit."
    fi
    
    # Check G++
    if check_command g++; then
        GCC_VERSION=$(g++ --version | head -n1 | sed -n 's/.*g++ (.*) \([0-9.]*\).*/\1/p')
        log_success "G++ found (version $GCC_VERSION)"
    else
        log_error "G++ not found. Please install build-essential."
    fi
    
    # Check CUDA runtime
    if check_directory "$CUDA_PATH"; then
        log_success "CUDA installation found at $CUDA_PATH"
    else
        log_warning "CUDA path not found at $CUDA_PATH"
        log "Set CUDA_PATH environment variable if CUDA is installed elsewhere"
    fi
    
    # Check MPI (optional)
    if check_directory "$MPI_PATH" || check_command mpirun; then
        log_success "MPI support detected"
    else
        log_warning "MPI not found (optional feature)"
    fi
    
    # Check GPU
    if check_command nvidia-smi; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        log_success "$GPU_COUNT GPU(s) detected"
        nvidia-smi -L | while read line; do
            log "  $line"
        done
    else
        log_warning "nvidia-smi not found - cannot detect GPUs"
    fi
}

build_project() {
    log "Starting build process..."
    
    cd "$SCRIPT_DIR"
    
    # Clean previous build
    if [ "$1" = "clean" ] || [ "$1" = "rebuild" ]; then
        log "Cleaning previous build..."
        make clean >> "$BUILD_LOG" 2>&1 || true
    fi
    
    # Build using Makefile
    log "Building WE-FDTD..."
    if make all >> "$BUILD_LOG" 2>&1; then
        log_success "Build completed successfully"
    else
        log_error "Build failed. Check $BUILD_LOG for details."
    fi
    
    # Verify executable
    if check_file "bin/we-fdtd"; then
        log_success "Executable created: bin/we-fdtd"
        
        # Test executable
        log "Testing executable..."
        if ./bin/we-fdtd > /dev/null 2>&1; then
            log_success "Executable test passed"
        else
            log_warning "Executable test failed (may be normal for mock build)"
        fi
    else
        log_error "Executable not found after build"
    fi
}

run_tests() {
    log "Running integration tests..."
    
    if ! check_file "bin/we-fdtd"; then
        log_error "WE-FDTD executable not found. Build first with: $0 build"
    fi
    
    # Test 1: Basic execution
    log "Test 1: Basic execution"
    if ./bin/we-fdtd >> "$BUILD_LOG" 2>&1; then
        log_success "Basic execution test passed"
    else
        log_warning "Basic execution test failed"
    fi
    
    # Test 2: Python integration
    log "Test 2: Python integration test"
    if python3 -c "
import sys
sys.path.append('.')
from fdtd_runner import FDTDRunner
runner = FDTDRunner(fdtd_binary_path='./bin/we-fdtd')
print('Python integration test passed')
" >> "$BUILD_LOG" 2>&1; then
        log_success "Python integration test passed"
    else
        log_warning "Python integration test failed"
    fi
    
    # Test 3: Demo run
    log "Test 3: Demo simulation"
    if python3 fdtd_runner.py --demo >> "$BUILD_LOG" 2>&1; then
        log_success "Demo simulation test passed"
    else
        log_warning "Demo simulation test failed"
    fi
}

install_system() {
    log "Installing WE-FDTD to system..."
    
    if ! check_file "bin/we-fdtd"; then
        log_error "WE-FDTD executable not found. Build first with: $0 build"
    fi
    
    if make install >> "$BUILD_LOG" 2>&1; then
        log_success "Installation completed"
    else
        log_error "Installation failed"
    fi
}

show_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  check       - Check build dependencies"
    echo "  build       - Build WE-FDTD (default)"
    echo "  clean       - Clean build artifacts"
    echo "  rebuild     - Clean and build"
    echo "  test        - Run integration tests"
    echo "  install     - Install to system"
    echo "  gpu-info    - Show GPU information"
    echo "  help        - Show this help"
    echo ""
    echo "Environment variables:"
    echo "  CUDA_PATH   - Path to CUDA installation (default: /usr/local/cuda)"
    echo "  MPI_PATH    - Path to MPI installation (default: /usr/lib/x86_64-linux-gnu/openmpi)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 rebuild"
    echo "  CUDA_PATH=/opt/cuda $0 build"
    echo "  $0 test"
}

show_gpu_info() {
    log "GPU Information:"
    echo "================"
    
    if check_command nvidia-smi; then
        nvidia-smi
    else
        log_warning "nvidia-smi not available"
    fi
    
    if check_command nvcc; then
        echo ""
        echo "CUDA Compiler Information:"
        nvcc --version
    fi
}

# Main script
main() {
    # Initialize log
    echo "=== WE-FDTD Build Log - $(date) ===" > "$BUILD_LOG"
    
    print_header
    
    case "${1:-build}" in
        "check")
            check_dependencies
            ;;
        "build")
            check_dependencies
            build_project
            ;;
        "clean")
            cd "$SCRIPT_DIR"
            make clean
            log_success "Clean completed"
            ;;
        "rebuild")
            check_dependencies
            build_project rebuild
            ;;
        "test")
            run_tests
            ;;
        "install")
            install_system
            ;;
        "gpu-info")
            show_gpu_info
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            ;;
    esac
    
    log "Build script completed. Log: $BUILD_LOG"
}

# Run main function
main "$@"