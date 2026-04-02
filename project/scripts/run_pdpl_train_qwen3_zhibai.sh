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

model_name=Qwen3-14B-Instruct
stage=sft
template=qwen3
Business=fangwusha

dataset=$1
gpus=$2

expname=${stage}-${template}-${Business}_${dataset}
output_dir=saves/${model_name}/${expname}
model_name_or_path=/group/40184/yougenyuan/tools/huggingface/Qwen3-14B

torchrun  --standalone --nnodes=1 --nproc-per-node=${gpus}  src/train.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${model_name_or_path} \
    --finetuning_type lora \
    --template ${template} \
    --enable_liger_kernel True \
    --flash_attn auto \
    --dataset_dir data \
    --dataset shenhe-250401-250531_juhe_qwen3_part1,shenhe-250401-250531_juhe_qwen3_part2,shenhe-250401-250531_juhe_qwen3_part3 \
    --deepspeed cache/ds_z3_config.json \
    --enable_thinking False \
    --cutoff_len 1024 \
    --learning_rate 1e-06 \
    --num_train_epochs 3.0 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_total_limit 50 \
    --save_steps 2000 \
    --max_steps 100000 \
    --warmup_steps 100 \
    --packing False \
    --report_to tensorboard \
    --output_dir ${output_dir} \
    --weight_decay 0.1 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
    


# --enable_liger_kernel \
