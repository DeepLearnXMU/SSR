import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM
import time
from tqdm import tqdm

choices = ["A", "B", "C", "D"]

llama2_template = \
'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] '''

alpaca_template = \
'''Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.

{instruction}

### Response:
'''


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1, template=None):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt if not template else template.format(instruction=prompt)


@torch.no_grad()
def _eval(args, subject, model, tokenizer, dev_df, test_df, template=None):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    if template:
        assert template in ['llama2', 'alpaca']
        template = eval(template+"_template")

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        if template:
            prompt = template.format(instruction=prompt)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            if template:
                prompt = template.format(instruction=prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        # print(prompt)
        # exit(-1)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids
            # decoder_input_ids=decoder_input_ids
        ).logits[:, -1, :].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[1]],
                        logits[tokenizer("B").input_ids[1]],
                        logits[tokenizer("C").input_ids[1]],
                        logits[tokenizer("D").input_ids[1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.lora_ckpt_dir:
        model = PeftModelForCausalLM.from_pretrained(model, args.lora_ckpt_dir)
    model = model.cuda()
    # heads_per_gpu = len(model.encoder.block) // args.ngpu
    # device_map = {
    #     gpu: list(
    #         range(
    #             0 + (gpu * heads_per_gpu),
    #             (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
    #         )
    #     )
    #     for gpu in range(args.ngpu)
    # }
    # model.parallelize(device_map)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.lora_ckpt_dir:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.lora_ckpt_dir))):
            os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.lora_ckpt_dir)))
    else:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
            os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for i, subject in enumerate(subjects):
        print(f'{i}/{len(subjects)} subject:', subject)
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = _eval(args, subject, model, tokenizer, dev_df, test_df, template=args.template)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.lora_ckpt_dir if args.lora_ckpt_dir else args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.lora_ckpt_dir if args.lora_ckpt_dir else args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.lora_ckpt_dir if args.lora_ckpt_dir else args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=2)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--lora_ckpt_dir", "-c", type=str, default="")
    parser.add_argument("--template", default="")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-small",
    )

    args = parser.parse_args()
    main(args)
