# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sftreg.trainer import CustomSeq2SeqTrainer
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
import os

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

def load_fisher(checkpoint_dir):
    # fisher = defaultdict(list)
    if not checkpoint_dir: return None
    if os.path.exists(os.path.join(checkpoint_dir[0], "fisher.pt")):
        return torch.load(os.path.join(checkpoint_dir[0], "fisher.pt"))
    return None

def run_sftreg(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    reg_cl_method: str,
    reg_p: float,
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    fisher = defaultdict(list)
    if reg_cl_method == "ewc":
        fisher = load_fisher(model_args.checkpoint_dir)
    episodic_mem = []
    if reg_cl_method != "ewc" or (reg_cl_method == "ewc" and fisher):
        model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
        
        optpar = defaultdict(list)
        for n, p in model.named_parameters():
            optpar[n] = torch.Tensor(p.cpu().data)
        # fisher = fisher.cuda()
        dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

        if training_args.predict_with_generate:
            tokenizer.padding_side = "left" # use left-padding in generation

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        )

        # Override the decoding parameters of Seq2SeqTrainer
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(
            generation_max_length=training_args.generation_max_length or data_args.max_target_length,
            generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
        ))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

        # Initialize our Trainer
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
            fisher=fisher,
            optpar=optpar,
            reg_cl_method=reg_cl_method,
            reg_p=reg_p,
            **split_dataset(dataset, data_args, training_args)
        )

        # Keyword arguments for `model.generate`
        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        if training_args.do_eval or training_args.do_predict:
            gen_kwargs["use_cache"] = True

        # Training
        if training_args.do_train:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            trainer.save_model()
            if trainer.is_world_process_zero() and model_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

        # Evaluation
        if training_args.do_eval:
            trainer.model.gradient_checkpointing_disable()
            trainer.model.config.use_cache = True
            metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
            if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
                metrics.pop("eval_loss", None)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if training_args.do_predict:
            trainer.model.gradient_checkpointing_disable()
            trainer.model.config.use_cache = True
            predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
            if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
                predict_results.metrics.pop("predict_loss", None)
            trainer.log_metrics("predict", predict_results.metrics)
            trainer.save_metrics("predict", predict_results.metrics)
            trainer.save_predictions(predict_results)
    else:
        fisher = defaultdict(list)
        print('Make fisher only...')
        print(model_args.checkpoint_dir, training_args.output_dir)
        model_args.checkpoint_dir = [training_args.output_dir]
        model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    
        dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

        if training_args.predict_with_generate:
            tokenizer.padding_side = "left" # use left-padding in generation

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        )

    # # create current task dataloader for EWC, save some training data of this task in the episodic memory
    # if training_args.cl_method == 'EWC':
    current_task_train_dataloader = DataLoader(
        dataset, shuffle=True, collate_fn=data_collator, batch_size=1
    )

    
    for idx_b, b in enumerate(current_task_train_dataloader):
        episodic_mem.append(b)
        if idx_b == 199: break
        # if idx_b == training_args.replay_num_instance_per_task: break
    # logger.info(f"----------- episodic_mem: {cl_model.episodic_mem} ---------")
    del current_task_train_dataloader

    ##### Compute Fisher info Matrix for EWC
    if reg_cl_method == "ewc": # training_args.cl_method == "EWC": # or training_args.cl_method == "L2":
        # model.gradient_checkpointing_disable()
        # model.config.use_cache = True
        model = model.cuda()
        fisher = defaultdict(list)
        # model = model.float()
        print("type:", type(model))
        # model.torch_dtype = torch.float
        # model.cpu() # to(dtype = torch.bfloat16).
        # print(model.dtype)
        for n, p in model.named_parameters():
            # cl_model.optpar[n] = torch.Tensor(p.cpu().data)
            fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()
        
        if reg_cl_method == "ewc": # training_args.cl_method == "EWC":
            for _, batch in tqdm(enumerate(episodic_mem)):
                model.zero_grad()
                model.eval()
                batch = {k: v.cuda() for k, v in batch.items()}
                # print(batch)
                # print(batch["input_ids"].dtype,batch["attention_mask"].dtype,batch["labels"].dtype)
                with torch.cuda.amp.autocast(): #
                    outputs = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
                loss = outputs.loss
                # logger.info(f"----------- loss: {loss} ---------")
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n].data += p.grad.data.cpu() ** 2

                # logger.info(f"----------- cl_model.fisher: {cl_model.fisher} ---------")

            for name_f,_ in fisher.items():
                fisher[name_f] /= len(episodic_mem) #*hparams.train_batch_size
            # model.zero_grad()
    
    torch.save(fisher, os.path.join(training_args.output_dir, "fisher.pt"))
    print('fisher.pt has been saved!')