#!/bin/bash
set -e
set -x

# parameter defalt values
tp=1
VLLM_DIR=/workspace/vllm-rocm
MODEL=/models/llama-2-7b-chat-hf/
BATCH_SIZE=1

# pring usage of the parameters
usage() {
    echo "Usage: $0 [--tp <n>] [--vllm_dir <path>] [--model <path>] [--batch-size <n>]"
    exit 1
}

# parse parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tp) tp="$2"; shift ;;
        --vllm_dir) VLLM_DIR="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift;;
        *) usage ;; # Any other argument will show usage information.
    esac
    shift # Move to next argument
done

# print parameter settings
echo "tensor parallel: $tp"
echo "vllm_dir: $VLLM_DIR"
echo "model: $MODEL"

# export DEBUG_CLR_GRAPH_PACKET_CAPTURE=1

ITER=5

array=(
    "896 1 128 128"
    "256 1 128 128"
    "120 1 128 2048"
    "84 1 2048 128"
    "64 1 2048 128"
    "16 1 2048 2048"
    "1 1 128 1"
    "1 1 2048 1"
)

cd $VLLM_DIR
for row in "${array[@]}"; do
    IFS=' ' read -r -a elements <<< "$row"
    bs=${elements[0]}
    tp=${elements[1]}
    input_len=${elements[2]}
    gen_len=${elements[3]}
    export TUNE_FP8=1
    echo "================================= TUNING $MODEL bs$bs tp$tp $input_len $gen_len ==============================================="
    torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size $bs --input-len $input_len --output-len $gen_len \
        --tensor-parallel-size $tp --quantization="fp8" --quantization-param-path="quark/llama.safetensors" --num-iters-warmup 0 --num-iters 1
    python3 gradlib/gradlib/fp8_gemm_tuner.py --input_file /tmp/fp8_shapes.csv --tuned_file /tmp/tuned_fp8_16.csv
    echo "================================= RUNNING $MODEL bs$bs tp$tp $input_len $gen_len ==============================================="
    torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size $bs --input-len $input_len --output-len $gen_len \
        --tensor-parallel-size $tp --quantization="fp8" --quantization-param-path="quark/llama.safetensors" --num-iters $ITER
done