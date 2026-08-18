nic_name="enp35s0f2"
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name  

export HCCL_BUFFSIZE=256
# export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false  
export OMP_NUM_THREADS=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
# export ASCEND_OPP_PATH=/home/zonghaoxin/installpkg/cann/opp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 # VLLM框架超时时间
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export TASK_QUEUE_ENABLE=1

export ASCEND_LAUNCH_BLOCKING=0

# for sparse attention
export DYNAMIC_EPLB="true"


# 确定性计算
# export LCCL_DETERMINISTIC=1 
# export CLOSE_MATMUL_K_SHIFT=1
# export HCCL_DETERMINISTIC=true
# export VLLM_ENABLE_V1_MULTIPROCESSING=0

# flash_comm
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
# mega moe
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_ASCEND_USE_BLOCKING_PP_P2P=0
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_WARMUP_PP_GROUP=1

#export COLLECT_LOGS_PATH=/home/zonghaoxin/plog/mega_moe1623
export COLLECT_LOGS_PATH=/home/g00893696/plog/plog_$(date +%Y%m%d_%H%M%S)
mkdir "$COLLECT_LOGS_PATH"
export ASCEND_PROCESS_LOG_PATH="$COLLECT_LOGS_PATH"
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1/0 是否打屏
export ASCEND_GLOBAL_LOG_LEVEL=1 # 0: debug 1: info 2: warning 3: error
#export PATH=/home/g00893696/ascendnpu-ir/tools/bishengir/bin:$PATH
# export DYNAMIC_EPLB="true"
export VLLM_PP_LAYER_PARTITION="4,8,8,8,8,8,8,8"

vllm serve /mnt/share/g00893696/MiniMax-M3-MXFP8 \
       --host 141.61.54.180 \
       --port 10086 \
       --served-model-name minimax \
       --trust-remote-code \
       --dtype bfloat16 \
       --max-num-seqs 64 \
       --max-num-batched-tokens 16384 \
       --max-model-len 262200 \
       --tensor-parallel-size 1 \
       --pipeline-parallel-size 8 \
       --data-parallel-size 1 \
       --quantization ascend \
       --gpu-memory-utilization 0.9 \
       --distributed_executor_backend "mp" \
       --no-enable-prefix-caching \
       --reasoning-parser minimax_m3 \
       --enable-expert-parallel \
       --safetensors-load-strategy 'prefetch' \
       --enforce-eager \
       --kv-cache-dtype fp8 \
       --no-async-scheduling \
       --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/g00893696/profiling", "torch_profiler_with_stack": false}'\
       --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false, "fuse_norm_quant":false}, "sparse_attn_decode_dtype":"bf16", "enable_gqa_kv_cache_fp8":true, "indexer_kv_dtype": "fp8", "multistream_overlap_shared_expert": true}' \
       > log1.log 2>&1 &

