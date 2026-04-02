# Activate conda environment to ensure python/torch is available
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-3.12.11

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export TOKENIZERS_PARALLELISM=False
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "

current_dir=$(pwd)
set -exo

#gpu_id="0"
# gpu_id="0,1,2,3"
gpu_id="0,1,2,3,4,5,6,7"

#gpu_id="0"
# gpu_id="0"
n=4
gpu_list="$gpu_id"
# 循环N-1次，每次拼接a前添加逗号
for ((i=2; i<=n; i++)); do
    gpu_list="${gpu_list},${gpu_id}"
done
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo $CHUNKS


dataset=$1
checkpoints=$2
stage=$3
stop_stage=$4

template=qwen3_5
Business=textevents

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"

    for checkpoint in ${checkpoints}; do
	for testset in largellm_260101_260126_v3_cv; do
	SPLIT=/apdcephfs_qy3/share_301069248/data/video/event_rag/merge/${testset}.jsonl
        expname=sft-${template}-${Business}_${dataset}
	model_name_or_path=/apdcephfs_qy3/share_301069248/huggingface/Qwen3.5-0.8B
        # model_name_or_path=/apdcephfs_hzlf/share_303924399/huggingface/Qwen3.5-0.8B-abliterated
        # model_name=saves/Qwen3.5-0.8B-abliterated/${expname}/checkpoint-${checkpoint}
	model_name=saves/Qwen3.5-0.8B/${expname}/checkpoint-${checkpoint}
        TARGET_DIR=${model_name}
        [ ! -d ${TARGET_DIR} ] && continue

        [ ! -f ${TARGET_DIR}/config.json ] && \
            llamafactory-cli export --model_name_or_path ${model_name_or_path} --adapter_name_or_path ${model_name} --template qwen3_5 --finetuning_type lora --export_dir ${model_name} --export_size 5 --export_device cpu --export_legacy_format false 
 
        max_new_token=100
        output_file=$TARGET_DIR/merge_${testset}_all2.jsonl
	[ -f ${output_file} ] && rm ${output_file}
        if [ ! -f ${output_file} ]; then
            for IDX in $(seq 0 $((CHUNKS-1))); do
            {
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python src/eval_textevents.py \
                    --model_name $model_name \
                    --dev_path $SPLIT \
                    --ans_file $TARGET_DIR/${testset}_${IDX}.jsonl \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX \
                    --max_new_token ${max_new_token}
            } &
            done
            wait

            echo $output_file
            # Clear out the output file if it exists.
            > "$output_file"
            # Loop through the indices and concatenate each file.
            for IDX in $(seq 0 $((CHUNKS-1))); do
                cat $TARGET_DIR/${testset}_${IDX}.jsonl | grep -v ": NaN" >> "$output_file"
            done
	fi 
        # python src/convert_vqav2_for_submission_textevents_debug.py  --split $SPLIT  --mejson $output_file  --precision 0.980 &> ${TARGET_DIR}/results_${testset}.log
        python src/convert_vqav2_for_submission_textevents_ragresults.py  --split $SPLIT  --mejson $output_file  --precision 0.980 &> ${TARGET_DIR}/results_${testset}.log
        done
    done
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"

    for checkpoint in ${checkpoints}; do
	# for testset in largellm_260101_260126_v3_cv; do
        for testset in keywords-230701-250931_v6_cv_juhe_qwen3; do 
	SPLIT=/apdcephfs_qy3/share_301069248/data/video/event_rag/merge/${testset}.jsonl
        expname=sft-${template}-${Business}_${dataset}
	model_name_or_path=/apdcephfs_qy3/share_301069248/huggingface/Qwen3.5-0.8B
        # model_name_or_path=/apdcephfs_hzlf/share_303924399/huggingface/Qwen3.5-0.8B-abliterated
        # model_name=saves/Qwen3.5-0.8B-abliterated/${expname}/checkpoint-${checkpoint}
	model_name=saves/Qwen3.5-0.8B/${expname}/checkpoint-${checkpoint}
        TARGET_DIR=${model_name}
        [ ! -d ${TARGET_DIR} ] && continue

        [ ! -f ${TARGET_DIR}/config.json ] && \
            llamafactory-cli export --model_name_or_path ${model_name_or_path} --adapter_name_or_path ${model_name} --template qwen3_5 --finetuning_type lora --export_dir ${model_name} --export_size 5 --export_device cpu --export_legacy_format false 
 
        max_new_token=100
        output_file=$TARGET_DIR/merge_${testset}_all2.jsonl
	[ -f ${output_file} ] && rm ${output_file}
        if [ ! -f ${output_file} ]; then
            for IDX in $(seq 0 $((CHUNKS-1))); do
            {
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python src/eval_textevents.py \
                    --model_name $model_name \
                    --dev_path $SPLIT \
                    --ans_file $TARGET_DIR/${testset}_${IDX}.jsonl \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX \
                    --max_new_token ${max_new_token}
            } &
            done
            wait

            echo $output_file
            # Clear out the output file if it exists.
            > "$output_file"
            # Loop through the indices and concatenate each file.
            for IDX in $(seq 0 $((CHUNKS-1))); do
                cat $TARGET_DIR/${testset}_${IDX}.jsonl | grep -v ": NaN" >> "$output_file"
            done
	fi 
        # python src/convert_vqav2_for_submission_textevents_debug.py  --split $SPLIT  --mejson $output_file  --precision 0.980 &> ${TARGET_DIR}/results_${testset}.log
        python src/convert_vqav2_for_submission_textevents_ragresults.py  --split $SPLIT  --mejson $output_file  --precision 0.980 &> ${TARGET_DIR}/results_${testset}.log
        done
    done
fi
