# from .task_list import *

from transformers import AutoTokenizer
import numpy as np
import jsonlines
import os
from tqdm import tqdm
from typing import List
import json


qa_task_list = ["task024_cosmosqa_answer_generation"]
qg_task_list = ["task074_squad1.1_question_generation"]
sa_task_list = ["task1312_amazonreview_polarity_classification"]
sum_task_list = ["task511_reddit_tifu_long_text_summarization"]
trans_task_list = ["task1219_ted_translation_en_es"]

dsg_task_list = ["task574_air_dialogue_sentence_generation"]
expl_task_list = ["task192_hotpotqa_sentence_generation"]
para_task_list = ["task177_para-nmt_paraphrasing"]
pe_task_list = ["task064_all_elements_except_first_i"]
pos_task_list = ["task346_hybridqa_classification"]


tokenizer = AutoTokenizer.from_pretrained("/home/hjh/data/llama2_7b_chat")
simsce_tokenizer = AutoTokenizer.from_pretrained("../hf_models/sup-simcse-roberta-base")


def len_filter(
    data,
    task_name: str,
    max_instruction_input_len_used=None,
    max_input_len_used=800,
    max_target_len_used=128,
):
    new_data = []
    stat_info = {}
    # value_counts = {}
    max_input_len, total_input_len, filtered_total_input_len = 0, 0, 0
    max_target_len, total_target_len, filtered_total_target_len = 0, 0, 0
    instruction_len = len(tokenizer.encode(data["Definition"][0]))
    for line in tqdm(data["Instances"], total=len(data["Instances"])):
        line["input"] = line["input"].strip()
        line["output"] = line["output"][0].strip()
        if line["input"] == "" or line["output"] == "":
            print("warning:", line)
            continue
        input_len = len(tokenizer.encode(line["input"]))
        target_len = len(tokenizer.encode(line["output"]))
        max_input_len = max(max_input_len, input_len)
        max_target_len = max(max_target_len, target_len)
        total_input_len += input_len
        total_target_len += target_len
        if input_len == 1 or target_len == 1:
            # print('Null Warning:', line)
            continue
        if (
            max_instruction_input_len_used
            and input_len + instruction_len > max_instruction_input_len_used
        ):
            continue
        if input_len > max_input_len_used or target_len > max_target_len_used:
            continue
        if (
            simsce_length := len(simsce_tokenizer.encode(data["Definition"][0]))
            + len(simsce_tokenizer.encode(line["input"]))
        ) > 510:
            print(
                "WARNING",
                simsce_length,
                instruction_len + input_len,
                instruction_len,
                input_len,
            )
            continue
        line["full_prompt"] = data["Definition"][0] + "\n\n" + line["input"]
        line["task_name"] = task_name

        # We verify that '\n\n' does not exist in all the task definitions.
        new_data.append(line)
        # if line['_task_name'] not in value_counts:
        #     value_counts[line['_task_name']] = 1
        # else:
        #     value_counts[line['_task_name']] += 1
        filtered_total_input_len += input_len
        filtered_total_target_len += target_len
    stat_info = {
        "task_name": task_name,
        "num": len(data["Instances"]),
        "instruction_len": instruction_len,
        "max_input_len": max_input_len,
        "max_target_len": max_target_len,
        "avg_input_len_ori": round(total_input_len / len(data["Instances"]), 4),
        "avg_target_len_ori": round(total_target_len / len(data["Instances"]), 4),
        "num_filtered": len(new_data),
        "avg_input_len": round(filtered_total_input_len / len(new_data), 4),
        "avg_target_len": round(filtered_total_target_len / len(new_data), 4),
    }
    return new_data, stat_info


niv2_dir = "./data/niv2/natural-instructions-2.8/tasks"

custom_dir = "./data/ni-cus0.12"

if not os.path.exists(custom_dir):
    os.makedirs(custom_dir)

if not os.path.exists(os.path.join(custom_dir, "filtered")):
    os.makedirs(os.path.join(custom_dir, "filtered"))

all_stat_info = []
for i, cate in enumerate(
    ["qa", "qg", "sa", "sum", "trans", "dsg", "para", "expl", "pe", "pos"]
):
    cate_data = []
    # cate_sample_data = []
    cate_task_list = eval(cate + "_task_list")
    for task in cate_task_list:
        print(cate + "\t" + task)
        with open(os.path.join(niv2_dir, task + ".json")) as f:
            data = json.load(f)
        new_data, stat_info = len_filter(data, task, max_instruction_input_len_used=500)
        print(stat_info)
        all_stat_info.append(stat_info)
        with jsonlines.open(
            os.path.join(custom_dir, "filtered", task + ".json"), "w"
        ) as f:
            f.write_all(new_data)

with jsonlines.open(os.path.join(custom_dir, "stat.json"), "w") as f:
    f.write_all(all_stat_info)
