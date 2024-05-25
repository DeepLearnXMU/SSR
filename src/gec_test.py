from transformers import AutoTokenizer
import transformers
import torch
import json
import os
from tqdm import tqdm
import copy

model = "/apdcephfs/share_733425/leyangcui/ptm/llama2-70b-chat-hf-bin/llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
# tokenizer.pad_token_id = tokenizer.eos_token_id
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id


# prompt =
# """<s>[INST] <<SYS>>
# {question} [/INST]
# """
prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Please fix obvious errors in the following sentences and keep the correct part unchanged. Noted that output should be formatted as a json format including two keys "corrected sentence" and "explanation": {question} [/INST]
"""



if __name__ == "__main__":
    batch_size = 100
    # question = "Fix obvious errors in the following sentences and keep the correct part unchanged."
    # question = "Fix obvious errors in the following sentences and keep the correct part unchanged. "
    input_path = "/apdcephfs/share_733425/leyangcui/gec/new_1bw/train_source"
    output_path = "/apdcephfs/share_733425/leyangcui/gec/new_1bw/train_llama70b_chat_output.txt"

    with open(input_path, 'r') as f:
        file = f.readlines()

    groups_of_10 = [file[i:i + batch_size] for i in range(0, len(file), batch_size)]
    for line in tqdm(groups_of_10[422+930:]):
        # input_query = question + line.strip()
        sequences = pipeline(
            [prompt.format(question=sent.strip()) for sent in line],
            do_sample=False,
            top_p=0.7,
            temperature=0.0,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=512,
            batch_size=50,
        )
        # import pdb
        # pdb.set_trace()
        # result = sequences[0]['generated_text'].split("[/INST]")[-1]
        result_text = [i[0]['generated_text'].split("[/INST]")[-1] for i in sequences]
        # result = result_text[0]
        # with open(output_path, "a", errors='ignore') as o:
        #     o.write(result.strip() + '\n')
        for input, output in zip(line, result_text):
            print(f'input:{input.strip()}')
            print(f'output:{output.strip()}')
            print('-------------')
            save_dict = {
                'input': input.strip(),
                'output': output.strip(),
            }
            with open("1bw_llama2_70b_decode_copy_2.jsonl", "a", encoding='utf-8') as file:
                json_str = json.dumps(save_dict, ensure_ascii=False)
                file.write(json_str + "\n")