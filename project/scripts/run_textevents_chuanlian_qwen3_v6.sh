set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=2
stop_stage=3

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    [ ! -f /nlp_group/README.md ] && mkdir /nlp_group && mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 11.186.41.79:/ /nlp_group
    for x in train_all_v6; do
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        GPUS=8
        PER_DEVICE_BATCH_SIZE=36
        GRADIENT_AC=2
	MAX_STEPS=20000
	SAVE_STEPS=250
	TASK_NAME=${x}
	MODEL_NAME="Qwen3-1.7B-Instruct"
	MODEL_NAME_OR_PATH="/apdcephfs_hzlf/share_303924399/huggingface/Qwen3-1.7B"
	FINETUNING_TYPE="full"
	TEMPLATE="qwen3"

	# DATASET="qunliaokey_241125-250313_juhe_qwen3,keyword_merge_all_251015_single_juhe_qwen3,shenhe-240601-250531_v4_keywords_juhe_qwen3,zhongzi-nlpsimilarity-250613_v2_keywords_juhe_qwen3,zhongzi-nlpsimilarity-250929_v2_keywords_juhe_qwen3"
	# DATASET="keywords-230701-250931_v1_train,shenhe-240601-250531_v4_keywords_llmverify,qunliaokey_241125-250313_juhe_qwen3,keyword_merge_all_251015_single_juhe_qwen3"
	# DATASET="keywords-230701-250931_v5_train,keywords-251001-251208_v5,qunliaokey_241125-250313_v2,shenhe-240601-250531_v5_keywords_llmverify,tongbao_chouyang_v2,zhongzi-nlpsimilarity-250613_v2_keywords_juhe_qwen3,zhongzi-nlpsimilarity-250929_v2_keywords_juhe_qwen3"
	# DATASET="rqunliaoke_qwen235b-process_0130,ragkeyword_260101_260110_processed_train_cot,Chinese-DeepSeek-R1-Distill-data-110k-SFT"
	# DATASET="Chinese-Qwen3-235B-2507-Distill-data-110k-SFT,keywords-251001-251208_v6_juhe_qwen3,keywords-230701-250931_v6_train_juhe_qwen3,qunliaokey_241125-250313_v3_juhe_qwen3,shenhe-240601-250531_v6_juhe_qwen3,rqunliaoke_qwen235b-process_0130,ragkeyword_260101_260110_processed_train_cot"
	# DATASET="Chinese-Qwen3-235B-2507-Distill-data-110k-SFT,rqunliaoke_qwen235b-process_0130,ragkeyword_260101_260110_processed_train_cot"

	DATASET="Chinese-Qwen3-235B-2507-Distill-data-110k-SFT,largellm_260101_260126_v3_train_juhe_qwen3,largellm_250801_251231_v1_juhe_qwen3,Infinity-Instruct_7M_core"
        # 使用环境变量传递参数并执行训练脚本
        GPUS=${GPUS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} GRADIENT_ACC=${GRADIENT_ACC} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
	TASK_NAME=${TASK_NAME} MODEL_NAME=${MODEL_NAME} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} FINETUNING_TYPE=${FINETUNING_TYPE} \
	TEMPLATE=${TEMPLATE} DATASET=${DATASET} \
        bash project/scripts/run_textevents_train_qwen3_2b_full_accelarate.sh
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v6; do
	# for checkpoint in 2356 14265; do
	for checkpoint in 3500; do
            bash project/scripts/run_textevents_eval_qwen3_2b_prompt1.sh ${x} ${checkpoint} 1 1
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v6; do
        for checkpoint in 3500; do
            bash project/scripts/run_textevents_eval_qwen3_2b_prompt1.sh ${x} ${checkpoint} 3 3
        done
    done
fi

echo "success on `date`"
# exit 0
