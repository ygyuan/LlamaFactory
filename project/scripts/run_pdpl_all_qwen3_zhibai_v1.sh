set -exo
current_dir=$(pwd)
cd ${current_dir}

stage=2
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v1; do
        gpus=8
        bash project/scripts/run_pdpl_train_qwen3_zhibai.sh ${x} ${gpus}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v1; do
        for checkpoint in 70000 50000 30000; do
            bash project/scripts/run_all_eval_qwen3_a100.sh ${x} ${checkpoint} 1 1
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    for x in train_all_v1; do
        for checkpoint in 480000; do
            stage=2
            stop_stage=2
            bash project/scripts/run_all_eval_qwen3.sh ${x} ${checkpoint} ${stage} ${stop_stage}
        done
    done
fi

echo "success on `date`"
exit 0
