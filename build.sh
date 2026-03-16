#!/bin/bash
# Build script for extension_cpp (CPU only)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Building extension_cpp (CPU only)"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Parse command line arguments
CLEAN_BUILD=false
DEBUG_BUILD=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            DEBUG_BUILD="1"
            shift
            ;;
        --verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean    Clean build directories before building"
            echo "  --debug    Build in debug mode (with -g -O0)"
            echo "  --verbose  Show verbose output"
            echo "  -h,--help  Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directories...${NC}"
    cd "$SCRIPT_DIR/extension_cpp"
    rm -rf build dist *.egg-info __pycache__ extension_cpp/__pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}Clean complete${NC}"
fi

# Set debug mode if requested
if [ -n "$DEBUG_BUILD" ]; then
    export DEBUG=1
    echo -e "${YELLOW}Building in DEBUG mode${NC}"
fi

# Build and install in development mode
echo ""
echo "Building and installing in development mode..."
cd "$SCRIPT_DIR/extension_cpp"

# Build with setup.py (for verbose output support)
if [ -n "$VERBOSE" ]; then
    python3 setup.py build_ext --verbose
else
    python3 setup.py build_ext
fi

# Install with pip (modern approach)
pip install -e . --no-build-isolation

echo ""
echo -e "${GREEN}======================================"
echo "Build completed successfully!"
echo "======================================${NC}"
echo ""
echo "You can now use the extension:"
echo "  python3 -c 'import torch; import extension_cpp.ops as ops; x=torch.range(-2, 2.5, 0.5); y=torch.empty_like(x); ops.mysilu_out(x, y); print(y)'"
echo ""
