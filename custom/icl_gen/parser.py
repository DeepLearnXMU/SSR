# Step 1 postprocess: parse ICL output

import jsonlines
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/hjh/data/hf_models/alpaca-7b-huggy")
simsce_tokenizer = AutoTokenizer.from_pretrained("../hf_models/sup-simcse-roberta-base")

max_instruction_input_len_used=500
max_simcse_raw_input_len_used=510
max_input_len_used=800 
max_target_len_used=128
save_more = True # only for alpaca-7b

for cate in ['qa', 'sum', 'qg', 'sa', 'trans']: #, 
    input_path = f'/home/hjh/data/public/SSR/data/ni-cus0.12/genearated-icl-naive/alpaca-7b/ori-van/{cate}.train.smp001.2shot.smp3.rp1.2.json'
    output_path = f'/home/hjh/data/public/SSR/data/ni-cus0.12/genearated-icl-naive-parsed-filtered/alpaca-7b/ori-van/{cate}.train.smp001.2shot.smp3.rp1.2.json'

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with jsonlines.open(input_path) as f:
        new_data = []
        for line in f:
            definition = line['inputs'].split('\n\nInput:')[0]
            instruction_len = len(tokenizer.encode(definition))
            insts = line['outputs'].split('Input:')
            for inst in insts:
                pair = inst.split('Output:')
                if len(pair) != 2:
                    # print(len(pair))
                    if save_more and len(pair) == 1:
                        ...
                    else:
                        continue
                if len(pair) == 2:
                    inputs, outputs = pair[0].strip(), pair[1].strip()
                else:
                    inputs, outputs = pair[0].strip(), ""

                input_len = len(tokenizer.encode(inputs))
                target_len = len(tokenizer.encode(outputs))
                
                if input_len == 1:# or target_len == 1:
                    # print('Null Warning:', line)
                    continue
                if target_len == 1 and not save_more:
                    continue
                if (
                    max_instruction_input_len_used
                    and input_len + instruction_len > max_instruction_input_len_used
                ):
                    continue
                if input_len > max_input_len_used or target_len > max_target_len_used:
                    continue
                if (
                    simsce_length := len(simsce_tokenizer.encode(inputs))
                    # len(simsce_tokenizer.encode(definition))
                    # + len(simsce_tokenizer.encode(inputs))
                ) > max_simcse_raw_input_len_used:
                    continue

                new_data.append({'inputs': definition + '\n\n' + inputs, 'outputs': outputs})

    with jsonlines.open(output_path, 'w') as f:
        f.write_all(new_data)