from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import sys
from tqdm import tqdm
import copy
import jsonlines

# import importlib
import ast
import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM


llama2_prompt = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST] """


alpaca_prompt = \
'''Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.

{prompt}

### Response:'''


def main(args):
    transformers.set_seed(42)

    # model = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.finetuning_type == "full":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path if not args.ckpt_dir else args.ckpt_dir)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            device="cuda"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model = PeftModelForCausalLM.from_pretrained(model, args.ckpt_dir)
        model = model.merge_and_unload()
        # model = model.cuda()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            device="cuda"
        )

    print("old model.config.use_cache:", pipeline.model.config.use_cache)
    pipeline.model.config.use_cache = True
    print("new model.config.use_cache:", pipeline.model.config.use_cache)


    pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

    input_path = args.input_path
    output_path = args.output_path

    with jsonlines.open(input_path, "r") as f:
        data = [line for line in f]

    if os.path.exists(output_path):
        print('Error: destination output path already exists!')
        exit(-1)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for i, line in tqdm(enumerate(data), total=len(data)):
        if 'inputs' in line:
            inputs = line['inputs']
        elif 'full_prompt' in line:
            inputs = line['full_prompt']
        else:
            inputs = line['definition_input']
        if args.template == "vanilla":
            instruction = inputs
        elif args.template == "llama2":
            instruction = llama2_prompt.format(prompt=inputs)
        elif args.template == "alpaca":
            instruction = alpaca_prompt.format(prompt=inputs)
        print("========")
        print(instruction)
        print("--------")
        sequences = pipeline(
            instruction,
            do_sample=args.do_sample,
            # top_p=args.top_p,
            # temperature=args.temperature,
            max_length=args.max_length,
            # min_new_tokens=,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            batch_size=1,
        )
        print(sequences)
        result_text = sequences[0]["generated_text"][len(instruction):].strip()

        save_dict = {
            "inputs": inputs,
            "targets": result_text,
        }

        with jsonlines.open(output_path, "a") as file:
            file.write(save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, required=True, help=""
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help=""
    )
    parser.add_argument("--input_path", type=str, default=None, required=True, help="")
    parser.add_argument("--output_path", type=str, default=None, required=True, help="")
    parser.add_argument("--do_sample", type=ast.literal_eval, default=False, help="")
    parser.add_argument("--top_p", type=float, default=0.6, help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="")
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--template", type=str, default="vanilla", help="")
    # parser.add_argument("--preserve_ratio", type=float, default=0.9, help="")
    # parser.add_argument("--preserve_word_step", type=int, default=1, help="")
    # parser.add_argument("--template_list_enum", type=ast.literal_eval, default=False)
    # parser.add_argument("--icl", type=ast.literal_eval, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
