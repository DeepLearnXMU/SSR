REPO_ROOT_DIR=~/data/LLaMA-Efficient-Tuning
SRC_DIR=$REPO_ROOT_DIR/unsioer/icl_gen
MODEL_DIR=/home/hjh/data/hf_models/llama-7b
# CATE="qa"
CKPT_DIR=$MODEL_DIR
CVD=${1:-0}
# GEN_CATE_LIST=${2:-}
N_SHOTS=${2:-2}
RETRIES=${3:-3}
RP=${4:-1.2}


echo do_sample_retries: $RETRIES


CUDA_VISIBLE_DEVICES=${CVD} python3 $SRC_DIR/complete_param_alpaca.py \
      --model_name_or_path $CKPT_DIR \
      --input_path "/home/hjh/data/LLaMA-Efficient-Tuning/data/alpaca_data_en_52k_smp50.json" \
      --output_path "/home/hjh/data/LLaMA-Efficient-Tuning/data/alpaca_data_en_52k_smp50.llama-7b.${N_SHOTS}shot.smp${RETRIES}.rp${RP}.json" \
      --do_sample True \
      --do_sample_retries $RETRIES \
      --top_p 0.6 \
      --temperature 0.9 \
      --repetition_penalty ${RP} \
      --max_length 2048 \
      --num_beams 1 \
      --n_shots ${N_SHOTS} \
      --template vanilla \
      --cate_task_style False
      # --resume True
