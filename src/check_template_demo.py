from llmtuner.extras.template import get_template_and_fix_tokenizer
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union
from transformers import AutoTokenizer

multiturn_examples = [
  {
    "instruction": "听起来很不错。人工智能可能在哪些方面面临挑战呢？",
    "input": "",
    "output": "人工智能面临的挑战包括数据隐私、安全和道德方面的问题，以及影响就业机会的自动化等问题。",
    "history": [
      ["你好，你能帮我解答一个问题吗？", "当然，请问有什么问题？"],
      ["我想了解人工智能的未来发展方向，你有什么想法吗？", "人工智能在未来的发展方向可能包括更强大的机器学习算法，更先进的自然语言处理技术，以及更加智能的机器人。"]
    ]
  },
  {
    "instruction": "好的，谢谢你！",
    "input": "",
    "output": "不客气，有其他需要帮忙的地方可以继续问我。",
    "history": [
      ["你好，能告诉我今天天气怎么样吗？", "当然可以，请问您所在的城市是哪里？"],
      ["我在纽约。", "纽约今天晴间多云，气温最高约26摄氏度，最低约18摄氏度，记得注意保暖喔。"]
    ]
  }
]

info = {
    "belle_multiturn": {
    "script_url": "belle_multiturn",
    "columns": {
      "prompt": "instruction",
      "query": "",
      "response": "output",
      "history": "history"
    }
  }
}

fixed_multiturn_examples = {
    "prompt": [],
    "query": [],
    "response": [],
    "history": []
}

for line in multiturn_examples:
    fixed_multiturn_examples['prompt'].append(line['instruction'])
    fixed_multiturn_examples['query'].append('')
    fixed_multiturn_examples['response'].append(line['output'])
    fixed_multiturn_examples['history'].append(line['history'])
    ...


tokenizer = AutoTokenizer.from_pretrained('/apdcephfs_cq2/share_1603164/data/huggingface_models/Llama-2-7b-hf')

vanilla_template = get_template_and_fix_tokenizer('vanilla', tokenizer)
llama2_template = get_template_and_fix_tokenizer('llama2', tokenizer)

max_source_length = 1024
max_target_length = 512
IGNORE_INDEX = -100


def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples["prompt"])):
        query, response = examples["prompt"][i], examples["response"][i]
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_supervised_dataset(template, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    max_length = max_source_length + max_target_length

    for query, response, history, system in construct_example(examples):
        input_ids, labels = [], []

        for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
            tokenizer, query, response, history, system
        )):
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length:
                target_ids = target_ids[:max_target_length]

            if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                break

            if turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


print('===vanilla_template===')
vanilla_mt_dataset = preprocess_supervised_dataset(vanilla_template, fixed_multiturn_examples)

print(vanilla_mt_dataset['input_ids'][0])
print(vanilla_mt_dataset['labels'][0])
print(tokenizer.convert_ids_to_tokens([id if id != -100 else 0 for id in vanilla_mt_dataset['input_ids'][0]]))
print(tokenizer.decode([id if id != -100 else 0 for id in vanilla_mt_dataset['input_ids'][0]]))
print(tokenizer.convert_ids_to_tokens([id if id != -100 else 0 for id in vanilla_mt_dataset['labels'][0]]))
print(tokenizer.decode([id if id != -100 else 0 for id in vanilla_mt_dataset['labels'][0]]))



print('===llama2_template===')
llama2_mt_dataset = preprocess_supervised_dataset(llama2_template, fixed_multiturn_examples)

# print(llama2_mt_dataset['labels'])

# def print_supervised_dataset_example(example):
#     print("input_ids:\n{}".format(example["input_ids"]))
#     print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
#     print("label_ids:\n{}".format(example["labels"]))
#     print("labels:\n{}".format(tokenizer.decode([
#         token_id if token_id != IGNORE_INDEX else tokenizer.pad_token_id for token_id in example["labels"]
#     ], skip_special_tokens=False)))
print(llama2_mt_dataset['input_ids'][0])
print(llama2_mt_dataset['labels'][0])
print(tokenizer.convert_ids_to_tokens([id if id != -100 else 0 for id in llama2_mt_dataset['input_ids'][0]]))
print(tokenizer.decode([id if id != -100 else 0 for id in llama2_mt_dataset['input_ids'][0]]))
print(tokenizer.convert_ids_to_tokens([id if id != -100 else 0 for id in llama2_mt_dataset['labels'][0]]))
print(tokenizer.decode([id if id != -100 else 0 for id in llama2_mt_dataset['labels'][0]]))



# dataset

# dataset = load_dataset(
#     data_path,
#     data_files=data_files,
#     split=split,
#     cache_dir=model_args.cache_dir,
#     streaming=streaming,
#     use_auth_token=None
# )