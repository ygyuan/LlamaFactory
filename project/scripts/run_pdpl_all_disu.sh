set -exo
current_dir=$(pwd)
cd ${current_dir}
#source ${current_dir}/.bashrc
#conda activate GPTSoVits
stage=2
stop_stage=2


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "start stage ${stage}, stop stage ${stop_stage}"
    scene_dir=/apdcephfs_qy3/share_301069248/data/video/pindaopinlun2
    # for x in pindaopl-250306-250306; do
    for x in pindaopl-shenhe-250401-250401; do
        input_data=${scene_dir}/${x}/${x}.txt
        split=${scene_dir}/${x}/${x}.jsonl
        [ ! -f ${split} ] && \
            python3 /apdcephfs_qy3/share_301069248/data/video/pindaopinlun2/local/data_prep_yyg.py ${input_data} ${split}

        output_dir=saves/models/disu_model_yunhandeng/checkpoint-highhit_normtext
        mkdir -p ${output_dir}
        mejson=${output_dir}/merge_${x}_all.jsonl
        [ ! -f ${mejson} ] && \
             CUDA_VISIBLE_DEVICES="3" python3 scripts_yyg/get_results_disu_qwen2_online2.py ${split} ${mejson}
        python3 src/convert_vqav2_for_submission_disu.py --mejson ${mejson} --split ${split} --precision 0.5
    done
fi
echo "success on `date`"
exit 0
