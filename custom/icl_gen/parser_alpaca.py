import jsonlines
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/hjh/data/hf_models/alpaca-7b-huggy")
simsce_tokenizer = AutoTokenizer.from_pretrained("../hf_models/sup-simcse-roberta-base")

max_instruction_input_len_used=500
max_simcse_raw_input_len_used=510
max_input_len_used=800 
max_target_len_used=128
# save_more = False # only for alpaca-7b

# for cate in ['qa', 'sum', 'qg', 'sa', 'trans']: #, 'dsg', 'expl', 'para', 'pe', 'pos']:
input_path = f'/home/hjh/data/public/SSR/data/alpaca_data_en_52k_smp50.llama-7b.2shot.smp3.rp1.2.json'
output_path = f'/home/hjh/data/public/SSR/data/alpaca/genearated-icl-naive-parsed-filtered/llama-7b/ori/alpaca_data_en_52k_smp50.llama-7b.2shot.smp3.rp1.2.json'

if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

with jsonlines.open(input_path) as f:
    new_data = []
    definition_input_list = []
    for line in f:
        pairs = line['outputs'].split('Output:')
        if len(pairs) != 2: continue
        definition_input, output = pairs[0], pairs[1].strip()
        if not output: continue
        input_pairs = definition_input.split('Input:')
        if len(input_pairs) != 2: continue
        # instruction_len = len(tokenizer.encode(definition))
        definition, input = input_pairs[0].strip(), input_pairs[1].replace('<noinput>', '').strip()
        if not definition:
            continue
        definition_input = definition + (('\n\n' + input) if input else '')
        if definition_input not in definition_input_list:
             definition_input_list.append(definition_input)
        else:
             continue
        if (
                simsce_length := len(simsce_tokenizer.encode(
                    definition_input
                ))
            ) > max_simcse_raw_input_len_used:
                continue
        # for inst in insts:
        #     pair = inst.split('Output:')
        #     if len(pair) != 2:
        #         # print(len(pair))
        #         if save_more and len(pair) == 1:
        #             ...
        #         else:
        #             continue
        #     if len(pair) == 2:
        #         inputs, outputs = pair[0].strip(), pair[1].strip()
        #     else:
        #         inputs, outputs = pair[0].strip(), ""

            # input_len = len(tokenizer.encode(inputs))
            # target_len = len(tokenizer.encode(outputs))
            
            # if input_len == 1:# or target_len == 1:
            #     # print('Null Warning:', line)
            #     continue
            # if target_len == 1 and not save_more:
            #     continue
            # if (
            #     max_instruction_input_len_used
            #     and input_len + instruction_len > max_instruction_input_len_used
            # ):
            #     continue
            # if input_len > max_input_len_used or target_len > max_target_len_used:
            #     continue
            # if (
            #     simsce_length := len(simsce_tokenizer.encode(inputs))
            #     # len(simsce_tokenizer.encode(definition))
            #     # + len(simsce_tokenizer.encode(inputs))
            # ) > max_simcse_raw_input_len_used:
            #     continue

        new_data.append({'definition_input': definition_input, 'definition': definition, 'input': input,  'output': output})

    with jsonlines.open(output_path, 'w') as f:
        f.write_all(new_data)