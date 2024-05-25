REPO_ROOT_DIR=/home/hjh/data/public/SSR
SRC_DIR=$REPO_ROOT_DIR/custom/icl_gen
MODEL_DIR=/home/hjh/data/llama2_7b
CATE=${1:-} # "qa"
CVD=${2:-0}
GEN_CATE_LIST=${3:-}
CATE_LIST=($GEN_CATE_LIST)

for CUR_CATE in ${CATE_LIST[@]};
do
   	echo $CUR_CATE
    CUDA_VISIBLE_DEVICES=${CVD} python3 $SRC_DIR/label_param.py \
        --model_name_or_path $MODEL_DIR \
        --ckpt_dir $REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B/lora/${CATE}/bs32x1x1-3ep-bf16 \
        --finetuning_type lora \
        --input_path $REPO_ROOT_DIR/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b/ori/$CUR_CATE.train.smp001.2shot.smp3.rp1.2.json \
        --output_path $REPO_ROOT_DIR/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b/${CATE}/$CUR_CATE.train.smp001.2shot.smp3.rp1.2.json \
        --do_sample False \
        --max_length 2048
done
