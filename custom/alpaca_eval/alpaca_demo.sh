export OPENAI_API_BASE=YOUR_OPENAI_API_BASE
export OPENAI_API_KEYS=YOUR_OPENAI_API_KEYS

alpaca_eval evaluate_from_model alpaca-7b-lora-cl-queue-alpaca001-rp01-trans -annotators_config chatgpt > alpaca_eval.alpaca-7b-lora-cl-queue-alpaca001-rp01-trans.chatgpt.result.log 2>&1 &
