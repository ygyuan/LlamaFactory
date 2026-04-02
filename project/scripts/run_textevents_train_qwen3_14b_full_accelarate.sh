export NCCL_P2P_DISABLE=1  # 禁用点对点通信
export NCCL_DEBUG=INFO  # 启用NCCL调试信息
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 按PCI总线ID排序GPU
export CUDA_LAUNCH_BLOCKING=1
export DISABLE_VERSION_CHECK=1 
export OMP_NUM_THREADS=1
export enable_sliding_window=False
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NODE_RANK=0
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_LAUNCH_BLOCKING=1
export MASTER_ADDR=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Define default values for variables
GPUS=${GPUS:-24}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=${GRADIENT_ACC:-2}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-10000}
TASK_NAME=${TASK_NAME:-'train_all_V2'}
MODEL_NAME=${MODEL_NAME:-'Qwen3-0.6B-Instruct'}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-'/group/40184/yougenyuan/tools/huggingface/Qwen3-14B'}
FINETUNING_TYPE=${FINETUNING_TYPE:-'lora'}
TEMPLATE=${TEMPLATE:-'qwen3'}
DATASET=${DATASET:-'shenhe-250401-250531_juhe_qwen3_part1'}

stage=sft
model_name=${MODEL_NAME}
Business=textevents
expname=${stage}-${TEMPLATE}-${Business}_${TASK_NAME}
output_dir=saves/${model_name}/${expname} && mkdir -p ${output_dir}


torchrun  --standalone --nnodes=1 --nproc-per-node=${GPUS}  src/train.py \
    --stage ${stage} \
    --do_train True \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --finetuning_type ${FINETUNING_TYPE} \
    --template ${TEMPLATE} \
    --enable_liger_kernel True \
    --enable_thinking False \
    --flash_attn auto \
    --deepspeed cache/ds_z3_config.json \
    --dataset_dir data_keywords \
    --dataset ${DATASET} \
    --trust_remote_code True \
    --cutoff_len 8196 \
    --learning_rate 1e-07 \
    --num_train_epochs 2.0 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 4 \
    --save_total_limit 400 \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps 25 \
    --packing False \
    --report_to tensorboard \
    --output_dir ${output_dir} \
    --weight_decay 0.1 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch
