
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export CC=/apdcephfs_qy3/share_301069248/users/aydentang/tool/gcc92/bin/gcc
export CXX=/apdcephfs_qy3/share_301069248/users/aydentang/tool/gcc92/bin/g++
#ixport LD_LIBRARY_PATH=::$LD_LIBRARY_PATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_DISABLED=true
export MASTER_PORT=34229
export NCCL_IB_DISABLE=1
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NODE_RANK=0
export NNODES=1
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export HF_DATASETS_CACHE="/apdcephfs_qy3/share_301069248/users/wenhangshi/llama_factory_dis2"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

#DS_CONFIG_PATH=examples/full_multi_gpu/ds_z2_offload_config.json
DS_CONFIG_PATH=examples/deepspeed/ds_z2_offload_config.json
OUTPUT_DIR=/apdcephfs_qy3/share_301069248/users/wenhangshi/LLaMA-Factory_1119/saves/kexin_lianwang/yitu/Qwen3-4B
OUTPUT_PATH=5k_1111



torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --model_name_or_path /apdcephfs_qy3/share_301069248/users/wenhangshi/UER/models/qwen3/Qwen3-4B \
    --dataset kexin_lianwang_yitu_5k_1111 \
    --dataset_dir data_TP \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir ${OUTPUT_DIR}/$OUTPUT_PATH \
    --ddp_timeout 9000 \
    --cutoff_len 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_strategy epoch \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --max_samples 10000 \
    --ddp_timeout 1800000 \
    --flash_attn fa2 \
    --bf16  \
    --enable_thinking False;







