#!/bin/bash
# Test script for extension_cpp

set -e

TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" TORCHINDUCTOR_CPP_WRAPPER=1 python3 example_mlp.py 2>&1 | tee log.txt