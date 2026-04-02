set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=2
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    [ ! -f /nlp_group/README.md ] && mkdir /nlp_group && mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 11.186.41.79:/ /nlp_group
    for x in train_all_v26; do
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
        GPUS=6
        PER_DEVICE_BATCH_SIZE=128
        GRADIENT_ACC=4
	MAX_STEPS=5000
	SAVE_STEPS=200
	TASK_NAME=${x}
	MODEL_NAME="Qwen3-0.6B-Instruct"
	MODEL_NAME_OR_PATH="/apdcephfs_hzlf/share_303924399/huggingface/Qwen3-0.6B"
	FINETUNING_TYPE="full"
	TEMPLATE="qwen3"

	# DATASET="qunliaokey_241125-250313_juhe_qwen3,keyword_merge_all_251015_single_juhe_qwen3,shenhe-240601-250531_v4_keywords_juhe_qwen3,zhongzi-nlpsimilarity-250613_v2_keywords_juhe_qwen3,zhongzi-nlpsimilarity-250929_v2_keywords_juhe_qwen3"
	# DATASET="keywords-230701-250931_v1_train,shenhe-240601-250531_v4_keywords_llmverify,qunliaokey_241125-250313_juhe_qwen3,keyword_merge_all_251015_single_juhe_qwen3"

	DATASET="keywords-230701-250931_v2_train,shenhe-240601-250531_v5_keywords_llmverify,tongbao_chouyang_v2,zhongzi-nlpsimilarity-250613_v3_keywords,zhongzi-nlpsimilarity-250929_v3_keywords"
        # 使用环境变量传递参数并执行训练脚本
        GPUS=${GPUS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} GRADIENT_ACC=${GRADIENT_ACC} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
	TASK_NAME=${TASK_NAME} MODEL_NAME=${MODEL_NAME} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} FINETUNING_TYPE=${FINETUNING_TYPE} \
	TEMPLATE=${TEMPLATE} DATASET=${DATASET} \
        bash project/scripts/run_pdpl_train_qwen3_zhibai_full_accelarate.sh
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v26; do
        # for checkpoint in 15000 12000 10000 8000 4000; do
	# for checkpoint in 18000 14000 10000; do
	# for checkpoint in 2500 2000 1500 1000 500; do
	for checkpoint in 1000 600; do
            bash project/scripts/run_all_eval_qwen3_06b_prompt1.sh ${x} ${checkpoint} 1 1
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v26; do
        for checkpoint in 3750; do
            bash project/scripts/run_all_eval_qwen3_06b_prompt1.sh ${x} ${checkpoint} 3 3
        done
    done
fi

echo "success on `date`"
exit 0
