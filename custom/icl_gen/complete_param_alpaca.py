from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import sys
from tqdm import tqdm
import copy
import jsonlines
from typing import List

# import importlib
import ast
import argparse



alpaca_prompt = \
'''Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.

{instruction}

### Response:'''

def times(n:int, length:int) -> List[List[int]]:
    '''
    n: number of shots.
    length: length of 
    '''
    if n == 1: return [[i] for i in range(length)]
    res_list = []
    for i in range(length):
        for lis in times(n-1, length):
            if i not in lis:
                res_list.append([i]+lis)
    return res_list


def pack_instructions(tokenizer, template, data, max_instances=10, perm_idx=None):
    n_shots = args.n_shots
    all_perm_list = times(n_shots, max_instances)
    inst_list, perm_list = [], []
    ignore = True
    for lis in all_perm_list:
        if perm_idx and ignore:
            if ignore and lis != perm_idx: continue
            else: 
                ignore = False
                continue
        instruction = 'Create task instructions following exmaples below.\n\n'
        for i in lis:
            input = data[i]['input']
            input = "<noinput>" if input.lower() == "" else input
            instruction += (
                f"Instruction: {data[i]['instruction']}\n" +
                f"Input: {input}\n" +
                f"Output: {data[i]['output']}\n\n")
        instruction += 'Instruction:'
        if template == "alpaca":
            instruction = alpaca_prompt.format(instruction=instruction)
        if len(tokenizer.tokenize(instruction))>=args.max_length:
            continue
        else:
            inst_list.append(instruction)
            perm_list.append(lis)

    return inst_list, perm_list

def icl_gen(pipeline, tokenizer, template, cur_data, max_instances, perm_idx=None):
    assert template in ["vanilla", "alpaca"]
    inst_list, perm_list = pack_instructions(tokenizer, template, cur_data, max_instances, perm_idx)
    for i, instruction in tqdm(enumerate(inst_list), total=len(inst_list)):
        if args.do_sample:
            sequences = []
            for _ in range(args.do_sample_retries):
                sequences.extend(pipeline(
                    instruction,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id,
                    batch_size=1,
                ))
        else:
            sequences = pipeline(
                instruction,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                max_length=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=1,
            )
        # print(sequences)
        result_texts = list(
            set(
                [
                    seq["generated_text"][len(instruction):]
                    for seq in sequences
                ]
            )
        )
        for text in result_texts:
            # print("======")
            # print(f"{cur_idx} / {len_words}")
            # print("------")
            # print(f"outputs:\n------\n{text}")
            # print("========")
            save_dict = {
                "inputs": instruction,
                "outputs": text,
                "perm_idx": perm_list[i]
                # "cur_idx": cur_idx,
                # "len_instruction_words": len_words,
            }

            with jsonlines.open(args.output_path, "a") as file:
                file.write(save_dict)


def main(args):
    transformers.set_seed(42)

    model = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device="cuda",
    )

    print("old model.config.use_cache:", pipeline.model.config.use_cache)
    pipeline.model.config.use_cache = True
    print("new model.config.use_cache:", pipeline.model.config.use_cache)


    pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

    input_path = args.input_path
    output_path = args.output_path

    with open(input_path, "r") as f:
        data = json.load(f)
    print('len(data):', len(data))
    if os.path.exists(output_path) and not args.resume:
        print('Error: destination output path already exists!')
        exit(-1)
    
    perm_idx = []
    if args.resume:
        with jsonlines.open(args.output_path, "r") as file:
            output_data = [l for l in file]
            perm_idx = output_data[-1]['perm_idx']
            print('perm_idx:', perm_idx)
            assert len(perm_idx)==args.n_shots

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
  
    icl_gen(pipeline, tokenizer, args.template, data, args.max_instances, perm_idx)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, required=True, help=""
    )
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--input_path", type=str, default=None, required=True, help="")
    parser.add_argument("--output_path", type=str, default=None, required=True, help="")
    parser.add_argument("--do_sample", type=ast.literal_eval, default=True, help="")
    parser.add_argument("--do_sample_retries", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.6, help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="")
    parser.add_argument("--n_shots", type=int, default=2)
    parser.add_argument("--template", type=str, default="vanilla", help="")
    parser.add_argument("--cate_task_style", type=ast.literal_eval, default=True)
    parser.add_argument("--resume", type=ast.literal_eval, default=False)
    # parser.add_argument("--preserve_ratio", type=float, default=0.9, help="")
    # parser.add_argument("--num_insert_loc", type=int, default=20, help="")
    # parser.add_argument("--template_list_enum", type=ast.literal_eval, default=False)
    # parser.add_argument("--icl", type=ast.literal_eval, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
