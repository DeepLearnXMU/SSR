REPO_ROOT_DIR=/home/hjh/data/public/SSR
SRC_DIR=$REPO_ROOT_DIR/custom/icl_gen
MODEL_DIR=/home/hjh/data/hf_models/alpaca-7b-huggy
# CATE="qa"
CKPT_DIR=$MODEL_DIR
CVD=${1:-0}
GEN_CATE_LIST=${2:-}
N_SHOTS=${3:-2}
RETRIES=${4:-3}
RP=${5:-1.2}


echo do_sample_retries: $RETRIES

for CUR_CATE in $GEN_CATE_LIST; # ${CATE_LIST[@]};
do
   echo $CUR_CATE
   CUDA_VISIBLE_DEVICES=${CVD} python3 $SRC_DIR/complete_param_nic010_cate.py \
         --model_name_or_path $CKPT_DIR \
         --input_path "${REPO_ROOT_DIR}/data/ni-cus0.12/split/$CUR_CATE.train.smp001.json" \
         --output_path "${REPO_ROOT_DIR}/data/ni-cus0.12/genearated-icl-naive/alpaca-7b/ori-van/$CUR_CATE.train.smp001.${N_SHOTS}shot.smp${RETRIES}.rp${RP}.json" \
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
done
