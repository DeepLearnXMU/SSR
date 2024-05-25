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
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM
from torch.utils.data import DataLoader


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


def pack_instructions(tokenizer, template, data, definition:str=None, cate_task_style:bool=True, perm_idx=None):
    n_shots = args.n_shots
    all_perm_list = times(n_shots, 4 if cate_task_style else 20)
    inst_list, perm_list = [], []
    ignore = True
    for lis in all_perm_list:
        if perm_idx and ignore:
            if ignore and lis != perm_idx: continue
            else: 
                ignore = False
                continue
        instruction = definition + '\n\n'
        for i in lis:
            instruction += ('Input: ' + data[i]['input'] + '\n' +
                'Output: ' + data[i]['output'] + '\n\n')
        instruction += 'Input:'
        if template == "llama2":
            instruction = llama2_prompt.format(prompt=instruction)
        if template == "alpaca":
            instruction = alpaca_prompt.format(prompt=instruction)
        if len(tokenizer.tokenize(instruction))>=args.max_length:
            continue
        else:
            inst_list.append(instruction)
            perm_list.append(lis)

    return inst_list, perm_list

class CustomDataLoader(DataLoader):
    ...

def icl_gen(pipeline, tokenizer, template, cur_data, definition, cate_task_style, perm_idx=None):
    assert template in ["vanilla", "llama2", "alpaca"]
    inst_list, perm_list = pack_instructions(tokenizer, template, cur_data, definition, cate_task_style, perm_idx)

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

    if os.path.exists(output_path) and not args.resume:
        print('Error: destination output path already exists!')
        exit(-1)
    
    perm_idx = []
    if args.resume:
        if os.path.exists(args.output_path):
            with jsonlines.open(args.output_path, "r") as file:
                output_data = [l for l in file]
                perm_idx = output_data[-1]['perm_idx']
                print('perm_idx:', perm_idx)
                assert len(perm_idx)==args.n_shots

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    if args.cate_task_style:
        for k in range(5):
            print('task', k)
            cur_data = data[k*4:k*4+4]

            definition = cur_data[0]['full_prompt'].split('\n\n')[0]
            assert definition == cur_data[-1]['full_prompt'].split('\n\n')[0]
            
            icl_gen(pipeline, tokenizer, args.template, cur_data, definition, args.cate_task_style)
    else:
        definition = data[0]['full_prompt'].split('\n\n')[0]
        assert definition == data[-1]['full_prompt'].split('\n\n')[0]    
        icl_gen(pipeline, tokenizer, args.template, data, definition, args.cate_task_style, perm_idx)
     

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
    parser.add_argument("--do_sample", type=ast.literal_eval, default=True, help="")
    parser.add_argument("--do_sample_retries", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.6, help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="")
    parser.add_argument("--finetuning_type", type=str, default="full")
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
