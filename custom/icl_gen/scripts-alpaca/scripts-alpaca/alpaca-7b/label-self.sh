REPO_ROOT_DIR=/home/hjh/data/LLaMA-Efficient-Tuning
SRC_DIR=$REPO_ROOT_DIR/unsioer/icl_gen
MODEL_DIR=/home/hjh/data/hf_models/alpaca-7b-huggy
# CATE=${1:-qa} # "qa"
CVD=${1:-0}
# GEN_CATE_LIST=${3:-qa}
CATE_LIST=($GEN_CATE_LIST)


CUDA_VISIBLE_DEVICES=${CVD} python3 $SRC_DIR/label_param.py \
    --model_name_or_path $MODEL_DIR \
    --input_path $REPO_ROOT_DIR/data/alpaca/genearated-icl-naive-kmeans520-self/llama-7b/ori/alpaca_data_en_52k_smp50.2shot.smp3.rp1.2.json \
    --output_path $REPO_ROOT_DIR/data/alpaca/genearated-icl-naive-kmeans520-self/llama-7b/ori-labeled/alpaca_data_en_52k_smp50.2shot.smp3.rp1.2.json \
    --do_sample False \
    --max_length 2048 \
    --template alpaca

