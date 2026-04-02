set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=1
stop_stage=1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    [ ! -f /nlp_group/README.md ] && mkdir /nlp_group && mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 11.186.41.79:/ /nlp_group
    for x in train_all_v13; do
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        GPUS=8
        PER_DEVICE_BATCH_SIZE=24
        GRADIENT_ACC=2
	MAX_STEPS=500000
	SAVE_STEPS=1000
	TASK_NAME=${x}
	MODEL_NAME_OR_PATH="/group/40106/yougenyuan/tools/huggingface/Qwen3-32B"
	FINETUNING_TYPE="lora"
	TEMPLATE="qwen3"
	DATASET="train_test_v10_remain_part3_juhe_qwen3"
	# DATASET="debug_juhe_qwen3"
	#DATASET="train_all_v4_juhe_qwen3,shenhe_zhongzi_all_v1_oov2_juhe_qwen3,zhongzi-all-250613_oov_juhe_qwen3"
	# DATASET="shenhe-repeat_v2_juhe_qwen3,shenhe-240601-250531_confuse_v2_juhe_qwen3,zhongzi-begin-250613_v2_juhe_qwen3,shenhe-250401-250531_v2_juhe_qwen3,shenhe-250201-250331_v2_juhe_qwen3,shenhe-241201-250131_v2_juhe_qwen3,shenhe-241001-241131_v2_juhe_qwen3,shenhe-240801-240931_v2_juhe_qwen3,shenhe-240701-240731_v2_juhe_qwen3,shenhe-240601-240631_v2_juhe_qwen3"
	#DATASET="shenhe-240601-250531_v2_repeat_juhe_qwen3,shenhe-240601-250531_v2_part1_juhe_qwen3,shenhe-240601-250531_v2_part2_juhe_qwen3,shenhe-240601-250531_v2_part3_juhe_qwen3,shenhe-240601-250531_v2_part4_juhe_qwen3,shenhe-240601-250531_v2_part5_juhe_qwen3,shenhe-240601-250531_v2_part6_juhe_qwen3,shenhe-240601-250531_v2_part7_juhe_qwen3,shenhe-240601-250531_v2_part8_juhe_qwen3,shenhe-240601-250531_v2_part9_juhe_qwen3,shenhe-240601-250531_v2_part10_juhe_qwen3"
        # 使用环境变量传递参数并执行训练脚本
        GPUS=${GPUS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} GRADIENT_ACC=${GRADIENT_ACC} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
	TASK_NAME=${TASK_NAME} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} FINETUNING_TYPE=${FINETUNING_TYPE} \
	TEMPLATE=${TEMPLATE} DATASET=${DATASET} \
        bash project/scripts/run_pdpl_train_qwen3_32b_zhibai_accelarate.sh
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v13; do
        for checkpoint in 20000; do
            bash project/scripts/run_all_eval_qwen3_32b.sh ${x} ${checkpoint} 1 1
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v13; do
        for checkpoint in 10000; do
            bash project/scripts/run_all_eval_qwen3_32b.sh ${x} ${checkpoint} 3 3
        done
    done
fi

echo "success on `date`"
exit 0
