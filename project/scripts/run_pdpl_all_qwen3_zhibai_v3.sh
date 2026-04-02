set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=3
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    [ ! -f /nlp_group/README.md ] && mkdir /nlp_group && mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 11.186.41.79:/ /nlp_group
    for x in train_all_v3_4B; do
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        GPUS=8
        PER_DEVICE_BATCH_SIZE=50
        GRADIENT_ACC=2
	MAX_STEPS=100000
	SAVE_STEPS=1000
	TASK_NAME=${x}
	MODEL_NAME_OR_PATH="/apdcephfs_qy3/share_301069248/users/yougenyuan/tools/huggingface/Qwen3-4B"
	FINETUNING_TYPE="lora"
	TEMPLATE="qwen3"
        DATASET="train_all_v4_juhe_qwen3,shenhe_zhongzi_all_v1_oov2_juhe_qwen3,zhongzi-all-250613_oov_juhe_qwen3"
        # 使用环境变量传递参数并执行训练脚本
        GPUS=${GPUS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} GRADIENT_ACC=${GRADIENT_ACC} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
	TASK_NAME=${TASK_NAME} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} FINETUNING_TYPE=${FINETUNING_TYPE} \
	TEMPLATE=${TEMPLATE} DATASET=${DATASET} \
        bash project/scripts/run_pdpl_train_qwen3_4b_zhibai_accelarate.sh
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v3_4B; do
        # for checkpoint in 20000 15000 10000 5000; do
	for checkpoint in 20000; do
            stage=1
            stop_stage=1
            bash project/scripts/run_all_eval_qwen3_4b_a100.sh ${x} ${checkpoint} ${stage} ${stop_stage}
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v3_4B; do
        for checkpoint in 480000; do
            stage=2
            stop_stage=2
            bash project/scripts/run_all_eval_qwen3_4b.sh ${x} ${checkpoint} ${stage} ${stop_stage}
        done
    done
fi

echo "success on `date`"
exit 0
