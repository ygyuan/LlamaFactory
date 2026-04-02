set -exo
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits
stage=2
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /apdcephfs_qy3/share_301069248/users/yougenyuan/tools/huggingface/Qwen2-1.5B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset alpaca_zh_demo \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2-1.5B-Instruct/full/train_2025-02-17-16-31-54 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
llamafactory-cli eval \
    --stage sft \
    --model_name_or_path /apdcephfs_qy3/share_301069248/users/yougenyuan/tools/huggingface/Qwen2-1.5B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --quantization_method bitsandbytes \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset alpaca_zh_demo \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen2-1.5B-Instruct/full/eval_2025-02-17-16-31-54 \
    --trust_remote_code True \
    --do_predict True
fi
echo "success on `date`"
exit 0



