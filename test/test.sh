#!/bin/bash
# Test script for extension_cpp

set -e

TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" TORCHINDUCTOR_CPP_WRAPPER=1 python3 example_mlp.py 2>&1 | tee log.txt

# TORCH_COMPILE_DEBUG=1 TORCH_LOGS="graph_breaks,inductor,aot,graph_code,output_code" python3 swiglu_mlp.py 2>&1 | tee log.txt