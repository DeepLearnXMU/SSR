import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer
from transformers.trainer import *

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        model = None,
        fisher = None,
        optpar = None,
        reg_cl_method = None,
        reg_p = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            *args,
            **kwargs,
        )
        self.fisher = fisher
        self.optpar = optpar
        self.reg_cl_method = reg_cl_method
        self.reg_p = reg_p

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = self._pad_tensors_to_target_len(
                        inputs["attention_mask"], inputs["labels"], pad_token_id=0
                    )
                if "position_ids" in inputs:
                    inputs["position_ids"] = self._pad_tensors_to_target_len(
                        inputs["position_ids"], inputs["labels"], pad_token_id=0
                    )

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        
        # logger.info(f"----------------- compute loss -----------------")


        ## origin T5
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        
        # print(f"------ current task id: {self.task_list_future.index(inputs['task_id'][0])} ------")
        # if self.reg_cl_method == "ADAPTERCL":
        #     outputs = model(
        #                 input_ids=inputs["input_ids"],
        #                 attention_mask=inputs["attention_mask"],
        #                 labels=inputs["labels"],
        #                 task_id=self.select_task_id_for_adapter(inputs),
        #                 # task_id=self.task_list_future.index(inputs["task_id"][0]),
        #                 # return_dict=False
        #             )
        # else:
        #     # inputs.pop('task_id')
        #     outputs = model(**inputs)
        outputs = model(**inputs)
        # print(f"------ LOSS: {outputs.loss} ------")


        if self.reg_cl_method == 'l2': # and not self.args.first_task:
            dev = next(model.parameters()).device
            l2_reg = 0

            for n,p in model.named_parameters():
                l = self.reg_p * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            loss = outputs.loss + l2_reg

        # elif self.reg_cl_method == "EWC" and not self.args.first_task:
        if self.reg_cl_method == "ewc":
            dev = next(model.parameters()).device
            ewc_loss = 0
            # reg = 0.01
            for n, p in model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                l = self.reg_p * self.fisher[n].to(dev) * (p - self.optpar[n].to(dev)).pow(2)
                ewc_loss += l.sum()

            # logger.info(f"================ ewc_loss={ewc_loss}")
            loss = outputs.loss + ewc_loss
        
        # elif self.reg_cl_method == 'AGEM' and not self.args.first_task:
        #     ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
        #     outputs.loss.backward(retain_graph=True)
        #     grad_cur = []
        #     for p in model.parameters():
        #         if p.requires_grad:
        #             grad_cur.append(p.grad.view(-1))
        #     grad_cur = torch.cat(grad_cur)
        #     # -check inequality constrain
        #     angle = (grad_cur * grad_ref).sum()
        #     if angle < 0:
        #         # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
        #         length_rep = (grad_ref * grad_ref).sum()
        #         grad_proj = grad_cur - (angle / length_rep) * grad_ref
        #         # -...and replace all the gradients within the model with this projected gradient
        #         index = 0
        #         for p in model.parameters():
        #             if p.requires_grad:
        #                 n_param = p.numel()  # number of parameters in [p]
        #                 p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
        #                 index += n_param

        #     # for AGEM, return the origin T5 loss as the return loss, but we don't need this loss, so we skip
        #     # the loss.backward() in the training_step()
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # else:

        #     ################################ HF default ################################
        #     # Save past state if it exists
        #     # TODO: this needs to be fixed and made cleaner later.
        #     if self.args.past_index >= 0:
        #         self._past = outputs[self.args.past_index]

        #     if labels is not None:
        #         if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #             loss = self.label_smoother(outputs, labels, shift_labels=True)
        #         else:
        #             loss = self.label_smoother(outputs, labels)
        #     else:
        #         if isinstance(outputs, dict) and "loss" not in outputs:
        #             raise ValueError(
        #                 "The model did not return a loss from the inputs, only the following keys: "
        #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #             )
        #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

