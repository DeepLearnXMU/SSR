REPO_ROOT_DIR=/home/hjh/data/public/SSR
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/home/hjh/data/hf_models/alpaca-7b-huggy
CUDA=${1:-0}
ORI_CKPT_DIR=""
CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/ALPACA-7B/lora/qa/bs32x1x1-3ep-bf16

CATE_LIST=("qa" "qg" "sa" "sum" "trans" "dsg" "expl" "para" "pe" "pos")

DATASETS=""

for CUR_CATE in ${CATE_LIST[@]}; 
do
    if [[ ${CATE_LIST[0]} != $CUR_CATE ]]
    then
        ORI_CKPT_DIR=$CKPT_DIR
    else
        continue
    fi
    CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/ALPACA-7B/lora/cl_queue_${CUR_CATE}/bs32x1x1-3ep-bf16
    if [ -d $CKPT_DIR ]; then
        echo File $CKPT_DIR exists.
        continue
    fi

    DATASETS="ni_c012_${CUR_CATE}_train"
    
    echo dataset: $DATASETS
    echo ORI_CKPT_DIR: $ORI_CKPT_DIR, CKPT_DIR: $CKPT_DIR
    CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
        --stage sft \
        --model_name_or_path $MODEL_DIR \
        --checkpoint_dir $ORI_CKPT_DIR \
        --do_train True \
        --overwrite_cache True \
        --finetuning_type lora \
        --template alpaca \
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
done
