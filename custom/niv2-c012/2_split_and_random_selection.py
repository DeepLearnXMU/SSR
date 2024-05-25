import numpy as np
import os
import jsonlines

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

custom_dir = "./data/ni-cus0.12"

if not os.path.exists(os.path.join(custom_dir, "split")):
    os.makedirs(os.path.join(custom_dir, "split"))

if not os.path.exists(os.path.join(custom_dir, "task-split")):
    os.makedirs(os.path.join(custom_dir, "task-split"))

num_train, num_eval, num_extra = 2000, 500, 20

all_stat_info = []
for i, cate in enumerate(
    ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]
):
    (
        cate_train_data,
        cate_eval_data,
        cate_extra_data,
        cate_smp001_data,
        cate_smp005_data,
        cate_smp01_data,
    ) = ([], [], [], [], [], [])
    cate_task_list = eval(cate + "_task_list")
    if cate not in ["sa", "pos"]:
        for task in cate_task_list:
            with jsonlines.open(
                os.path.join(custom_dir, "filtered", task + ".json")
            ) as f:
                data = [line for line in f]
            np.random.seed(0)
            sample_id_list = np.random.choice(
                len(data), num_train + num_eval + num_extra, replace=False
            )
            train_id_list, eval_id_list, extra_id_list = (
                sample_id_list[:num_train],
                sample_id_list[num_train:(num_train+num_eval)],
                sample_id_list[-num_extra:]
            )
            train_data, eval_data, extra_data = [], [], []

            for id in train_id_list:
                train_data.append(data[id])
            for id in eval_id_list:
                eval_data.append(data[id])
            for id in extra_id_list:
                extra_data.append(data[id])

            train_smp01_data = train_data[: num_train // 10]
            train_smp005_data = train_data[: num_train // 20]
            train_smp001_data = train_data[-num_train // 100 :]
            cate_train_data.extend(train_data)
            cate_eval_data.extend(eval_data)
            cate_extra_data.extend(extra_data)
            cate_smp01_data.extend(train_smp01_data)
            cate_smp005_data.extend(train_smp005_data)
            cate_smp001_data.extend(train_smp001_data)
    else:
        # binary classification
        pos_labels = ["positive", "true"]
        neg_labels = ["negative", "false"]
        for task in cate_task_list:
            with jsonlines.open(
                os.path.join(custom_dir, "filtered", task + ".json")
            ) as f:
                data = [line for line in f]
            pos_data, neg_data = [], []
            for line in data:
                if line["output"].lower() in pos_labels:
                    pos_data.append(line)
                elif line["output"].lower() in neg_labels:
                    neg_data.append(line)
                else:
                    print("warning: unknown labels", line)

            np.random.seed(0)
            pos_sample_id_list = np.random.choice(
                len(pos_data), (num_train + num_eval + num_extra) // 2, replace=False
            )
            neg_sample_id_list = np.random.choice(
                len(neg_data), (num_train + num_eval + num_extra) // 2, replace=False
            )
            pos_train_id_list, pos_eval_id_list, pos_extra_id_list = (
                pos_sample_id_list[: num_train // 2],
                pos_sample_id_list[num_train // 2 :(num_train+num_eval)//2],
                pos_sample_id_list[-num_extra // 2 :]
            )
            neg_train_id_list, neg_eval_id_list, neg_extra_id_list = (
                neg_sample_id_list[: num_train // 2],
                neg_sample_id_list[num_train // 2 :(num_train+num_eval)//2],
                neg_sample_id_list[-num_extra // 2 :]
            )
            train_data, eval_data, extra_data = [], [], []
            for id in pos_train_id_list:
                train_data.append(pos_data[id])
            for id in pos_eval_id_list:
                eval_data.append(pos_data[id])
            for id in pos_extra_id_list:
                extra_data.append(pos_data[id])
            for id in neg_train_id_list:
                train_data.append(neg_data[id])
            for id in neg_eval_id_list:
                eval_data.append(neg_data[id])
            for id in neg_extra_id_list:
                extra_data.append(neg_data[id])
            train_smp01_data = (
                train_data[: num_train // 20]
                + train_data[num_train // 2 : num_train // 2 + num_train // 20]
            )
            train_smp005_data = (
                train_data[: num_train // 40]
                + train_data[num_train // 2 : num_train // 2 + num_train // 40]
            )
            train_smp001_data = (
                train_data[-num_train // 200 :]
                + train_data[num_train // 2 - num_train // 200 : num_train // 2]
            )
            cate_train_data.extend(train_data)
            cate_eval_data.extend(eval_data)
            cate_extra_data.extend(extra_data)
            cate_smp01_data.extend(train_smp01_data)
            cate_smp005_data.extend(train_smp005_data)
            cate_smp001_data.extend(train_smp001_data)

    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".train.json"), "w"
    ) as f:
        f.write_all(cate_train_data)
    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".eval.json"), "w"
    ) as f:
        f.write_all(cate_eval_data)
    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".extra.json"), "w"
    ) as f:
        f.write_all(cate_extra_data)
    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".train.smp01.json"), "w"
    ) as f:
        f.write_all(cate_smp01_data)
    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".train.smp005.json"), "w"
    ) as f:
        f.write_all(cate_smp005_data)
    with jsonlines.open(
        os.path.join(custom_dir, "split", cate + ".train.smp001.json"), "w"
    ) as f:
        f.write_all(cate_smp001_data)
