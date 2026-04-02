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
#gpu_id="0,1,2,3,4,5,6,7"

#gpu_id="0"
gpu_id="0,1,2,3,4,5"
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

template=qwen3
Business=fangwusha

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"

    for checkpoint in ${checkpoints}; do
        # for testset in test_v3 keywords-250925-250925 shenhe-250601-250601_v2; do
	# for testset in test_v3 keywords-250925-250925 keywords-230701-250931_v1_cv; do
        # for testset in keywords-251210-251210 test_qunliao_keywords_v1 keywords-230701-250931_v5_cv; do
        for testset in keywords-230701-250931_v5_cv; do
	SPLIT=/apdcephfs_qy3/share_301069248/data/video/qunliao/merge/${testset}_juhe_qwen3.jsonl
 
        model_name=/apdcephfs_qy3/share_301069248/users/yougenyuan/handover/EasyR1/checkpoints/easy_r1/${dataset}/global_step_${checkpoint}/actor/huggingface
        TARGET_DIR=${model_name}
        [ ! -f ${TARGET_DIR}/model.safetensors ] && \
            python3 /apdcephfs_qy3/share_301069248/users/yougenyuan/handover/EasyR1/scripts/model_merger.py --local_dir ${TARGET_DIR}

        max_seq_len=4096
        max_new_token=2048
        output_file=$TARGET_DIR/merge_${testset}_all.jsonl
        if [ ! -f ${output_file} ]; then
            for IDX in $(seq 0 $((CHUNKS-1))); do
            {
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python src/eval_zhibai_grpo.py \
                    --model_name $model_name \
                    --dev_path $SPLIT \
                    --ans_file $TARGET_DIR/${testset}_${IDX}.jsonl \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX \
                    --max_seq_len ${max_seq_len} \
                    --max_new_token ${max_new_token}
            }
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
        python src/convert_vqav2_for_submission_zhibai_debug.py  --split $SPLIT  --mejson $output_file  --precision 0.990 &> ${TARGET_DIR}/results_${testset}.log
        done
    done
fi
