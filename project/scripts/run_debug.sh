set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=1
stop_stage=1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    [ ! -f /nlp_group/README.md ] && mkdir /nlp_group && mount -t nfs -o vers=3,nolock,proto=tcp,noresvport 11.186.41.79:/ /nlp_group
    for x in debug; do
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
        # export CUDA_VISIBLE_DEVICES="0"
        GPUS=6
        PER_DEVICE_BATCH_SIZE=2
        GRADIENT_ACC=2
	MAX_STEPS=10000000
	SAVE_STEPS=250000
	TASK_NAME=${x}
	#MODEL_NAME_OR_PATH="/apdcephfs_qy3/share_301069248/users/yougenyuan/tools/huggingface/Qwen3-14B"
        #MODEL_NAME_OR_PATH="/dockerdata/huggingface/Qwen3-14B"
	# MODEL_NAME_OR_PATH="/apdcephfs_hzlf/share_303924399/huggingface/Qwen3-14B"
	MODEL_NAME_OR_PATH="/apdcephfs_qy3/share_301069248/huggingface/Qwen3-14B"
	FINETUNING_TYPE="lora"
	TEMPLATE="qwen3"
        DATASET="debug_juhe_qwen3"
	#DATASET="shenhe-240601-250531_v3_confuse_juhe_qwen3,zhongzi-begin-250613_v2_juhe_qwen3,shenhe-250401-250531_v2_juhe_qwen3"
	#DATASET="shenhe-240601-250531_v2_repeat_juhe_qwen3,shenhe-240601-250531_v2_part1_juhe_qwen3,shenhe-240601-250531_v2_part2_juhe_qwen3,shenhe-240601-250531_v2_part3_juhe_qwen3,shenhe-240601-250531_v2_part4_juhe_qwen3,shenhe-240601-250531_v2_part5_juhe_qwen3,shenhe-240601-250531_v2_part6_juhe_qwen3,shenhe-240601-250531_v2_part7_juhe_qwen3,shenhe-240601-250531_v2_part8_juhe_qwen3,shenhe-240601-250531_v2_part9_juhe_qwen3,shenhe-240601-250531_v2_part10_juhe_qwen3"
        # 使用环境变量传递参数并执行训练脚本
        GPUS=${GPUS} PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} GRADIENT_ACC=${GRADIENT_ACC} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS}\
	TASK_NAME=${TASK_NAME} MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} FINETUNING_TYPE=${FINETUNING_TYPE} \
	TEMPLATE=${TEMPLATE} DATASET=${DATASET} \
        bash project/scripts/run_pdpl_train_qwen3_zhibai_accelarate.sh
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in debug; do
        for checkpoint in 48000 42000 36000 30000 24000 18000 12000 6000; do
            stage=1
            stop_stage=1
            bash scripts_yyg/run_pdpl_eval_qwen2_5_jiangliang.sh ${x} ${checkpoint} ${stage} ${stop_stage}
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in debug; do
        for checkpoint in 480000; do
            stage=2
            stop_stage=2
            bash scripts_yyg/run_pdpl_eval_qwen2_5_jiangliang.sh ${x} ${checkpoint} ${stage} ${stop_stage}
        done
    done
fi

echo "success on `date`"
exit 0
