REPO_ROOT_DIR=/home/hjh/data/public/SSR
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/home/hjh/data/llama2_7b_chat/
CUDA=${1:-0}
CKPT_DIR=""

CATE_LIST=("qa" "qg" "sa" "sum" "trans" "dsg" "expl" "para" "pe" "pos")

DATASETS=""

for CUR_CATE in "pos"; # ${CATE_LIST[@]}; 
do
    if [[ ${CATE_LIST[0]} != $CUR_CATE ]]
    then
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/cl_queue_${CUR_CATE}/bs32x1x1-3ep-bf16
    else
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/${CUR_CATE}/bs32x1x1-3ep-bf16
    fi

    DATASETS="ni_c012_${CUR_CATE}_train"

    for CUR_CATE_Y in ${CATE_LIST[@]};
    do
        echo $CUR_CATE_Y
        if [ -f $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval/all_results.json ]; then
            echo File $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval/all_results.json exists.
            cat $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval/all_results.json | grep rouge-l
            continue
        fi
        CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
            --stage sftrp \
            --model_name_or_path $MODEL_DIR \
            --checkpoint_dir $CKPT_DIR \
            --overwrite_cache True \
            --predict_with_generate True \
            --finetuning_type lora \
            --template llama2 \
            --dataset_dir $REPO_ROOT_DIR/data/ \
            --dataset ni_c012_${CUR_CATE_Y}_eval \
            --max_source_length 1024 \
            --max_target_length 512 \
            --max_samples 100000 \
            --per_device_eval_batch_size 1 \
            --output_dir $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval \
            --do_predict True \
            --do_sample False \
            --bf16 True     
    done
done
