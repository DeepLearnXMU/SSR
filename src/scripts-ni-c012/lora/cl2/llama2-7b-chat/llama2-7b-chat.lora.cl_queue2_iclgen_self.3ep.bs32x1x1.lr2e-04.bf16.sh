REPO_ROOT_DIR=/home/hjh/data/public/SSR
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/home/hjh/data/llama2_7b_chat/
CUDA=${1:-0}
ORI_CKPT_DIR=""

CATE_LIST=("trans" "sa" "qa" "sum" "qg") # "dsg" "expl" "para" "pe" "pos")
CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/${CATE_LIST[0]}/bs32x1x1-3ep-bf16

DATASETS=""

for CUR_CATE in ${CATE_LIST[@]}; 
do
    # echo CATE_LIST[0]: ${CATE_LIST[0]}
    if [[ ${CATE_LIST[0]} != $CUR_CATE ]]
    then
        ORI_CKPT_DIR=$CKPT_DIR
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/cl_queue2_iclgen_self_${CUR_CATE}/bs32x1x1-3ep-bf16
        echo $CKPT_DIR

        if [ -d $CKPT_DIR ]; then
            echo File $CKPT_DIR exists.
        else
            DATASETS="ni_c012_${CUR_CATE}_train"
            for PREV_CATE in ${CATE_LIST[@]};
            do
                if [[ $PREV_CATE == $CUR_CATE ]]
                then
                    break
                else
                    DATASETS="$DATASETS,ni_c012_icl_gen_km20_self_cl_queue2_llama2_7b_chat_${PREV_CATE}"
                fi
            done
            
            echo dataset: $DATASETS
            echo ORI_CKPT_DIR: $ORI_CKPT_DIR, CKPT_DIR: $CKPT_DIR
            CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
                --stage sft \
                --model_name_or_path $MODEL_DIR \
                --checkpoint_dir $ORI_CKPT_DIR \
                --do_train True \
                --overwrite_cache True \
                --finetuning_type lora \
                --template llama2 \
                --dataset_dir $REPO_ROOT_DIR/data/ \
                --dataset $DATASETS \
                --max_source_length 1024 \
                --max_target_length 512 \
                --learning_rate 2e-04 \
                --num_train_epochs 3.0 \
                --max_samples 100000 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 1 \
                --lr_scheduler_type cosine \
                --max_grad_norm 1.0 \
                --logging_steps 5 \
                --save_strategy no \
                --warmup_steps 0 \
                --lora_rank 8 \
                --lora_dropout 0.1 \
                --lora_target q_proj,v_proj \
                --resume_lora_training True \
                --output_dir $CKPT_DIR \
                --plot_loss True \
                --bf16 True
            fi
    fi

    OUTPUT_PATH=$REPO_ROOT_DIR"/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b-chat/cl_queue2/$CUR_CATE.train.smp001.2shot.smp3.rp1.2.json"

    if [ ! -f $OUTPUT_PATH ]; then
        echo File $OUTPUT_PATH does not exist.

        CUDA_VISIBLE_DEVICES=$CUDA python3 $REPO_ROOT_DIR/custom/icl_gen/label_param.py \
            --model_name_or_path $MODEL_DIR \
            --ckpt_dir $CKPT_DIR \
            --finetuning_type lora \
            --input_path $REPO_ROOT_DIR/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b-chat/ori-van/$CUR_CATE.train.smp001.2shot.smp3.rp1.2.json \
            --output_path $OUTPUT_PATH \
            --do_sample False \
            --max_length 2048 \
            --template llama2
    else
        echo File $OUTPUT_PATH exists.
    fi
    
done

for CUR_CATE in ${CATE_LIST[@]}; 
do
    echo CUR_CATE: $CUR_CATE
    if [[ ${CATE_LIST[0]} != $CUR_CATE ]]
    then
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/cl_queue2_iclgen_self_${CUR_CATE}/bs32x1x1-3ep-bf16
    else
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B-Chat/lora/${CUR_CATE}/bs32x1x1-3ep-bf16
    fi

    DATASETS="ni_c012_${CUR_CATE}_train"

    for CUR_CATE_Y in ${CATE_LIST[@]};
    do
        echo CUR_CATE_Y: $CUR_CATE_Y
        if [ -d $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval ]; then
            echo File $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval exists.
            continue
        else
            echo $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval does not exist.
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
