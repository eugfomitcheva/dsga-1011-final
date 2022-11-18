import torch, gc
gc.collect()
torch.cuda.empty_cache()

from datasets import load_dataset
from datasets import get_dataset_split_names
from model_vv import ModelMultitaskBinary
from moe_vv import *
import wandb
wandb.login()   


import argparse
import torch
import torch.nn as nn
from dataset import *
from transformers import PegasusForConditionalGeneration,PegasusTokenizer
from model_vv import ModelMultitaskBinary
from training_utils import *


parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
parser.add_argument('--expert_hidden_size', type=int, default=1024)
parser.add_argument('--tower_hidden_size', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=1024) # 768 / 1024
parser.add_argument('--bottom_hidden_size', type=int, default=1024)
parser.add_argument('--num_experts', type=int, default=6)
parser.add_argument('--scoring_methods', type=str, default = ["rouge_1","rogue_2","rouge_l"])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--max_len', type=int, default=448)
parser.add_argument('--max_summ_len', type=int, default=64)


args = parser.parse_args("")
args.n_tasks = len(args.scoring_methods)
args.n_positives = 1
args.n_negatives = 1


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args.device = device

from transformers import Trainer, TrainingArguments, default_data_collator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from utils import *

class CustomTrainer(Trainer):
    def nested_detach(tensors):
        if isinstance(tensors, (list, tuple)):
            return type(tensors)(nested_detach(t) for t in tensors)
        return tensors.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        mode = inputs["mode"]
        text_and_summaries_ids = inputs["text_and_summaries_input_ids"]
        text_and_summaries_mask = inputs["text_and_summaries_attn_mask"]
        scores = inputs["scores"]

        outputs = model(mode, text_and_summaries_ids, text_and_summaries_mask, scores)

        loss = outputs["loss"]
        output = torch.zeros(2 + 3 * args.n_tasks + 2).float().to(loss.device)
        output[0] = loss
        output[1] = outputs["loss_nce"]
        for j in range(args.n_tasks):
            output[2 + j * 3] = outputs["accuracy_{}".format(args.scoring_methods[j])]
            output[3 + j * 3] = outputs["rank_{}".format(args.scoring_methods[j])]
            output[4 + j * 3] = outputs["prediction_{}".format(args.scoring_methods[j])]
        output[-2] = outputs["prediction_sum"]
        output[-1] = outputs["overall_sum"]
        if(torch.sum(mode)) <= 0:
            wandb.log({'r1': outputs["r1"]}, commit=False)
            wandb.log({'r2': outputs["r2"]}, commit=False)
            wandb.log({'rl': outputs["rl"]}, commit=False)

        
        return (loss, output) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    # with autocast():
                    outputs = model(**inputs)
                else:
                    text_inputs_ids = inputs["text_inputs_ids"]
                    text_attention_mask = inputs["text_attention_mask"]
                    text_inputs = {
                        "input_ids": text_inputs_ids,
                        "attention_mask": text_attention_mask
                    }
                    outputs = model(**text_inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    loss_nce = np.mean([preds[i] for i in range(0, len(preds), 1 + 3 * args.n_tasks + 2)])
    result = {
        "loss_nce": loss_nce
    }
    for j in range(args.n_tasks):
        accuracy_arr = [preds[i] for i in range(1 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        accuracy = np.mean(accuracy_arr)
        rank_arr = [preds[i] for i in range(2 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        rank = np.mean(rank_arr)
        prediction_arr = [preds[i] for i in range(3 + j * 3, len(preds), 1 + 3 * args.n_tasks + 2)]
        prediction = np.mean(prediction_arr)
        print("Task {}, # pred batches: {}".format(j + 1, len(accuracy_arr)))
        result["accuracy_{}".format(args.scoring_methods[j])] = accuracy
        result["rank_{}".format(args.scoring_methods[j])] = rank
        result["prediction_{}".format(args.scoring_methods[j])] = prediction
    prediction_sum = np.mean([preds[i] for i in range(1 + 3 * args.n_tasks, len(preds), 1 + 3 * args.n_tasks + 2)])
    result["prediction_sum"] = prediction_sum
    overall_sum = np.mean([preds[i] for i in range(1 + 3 * args.n_tasks + 1, len(preds), 1 + 3 * args.n_tasks + 2)])
    result["overall_sum"] = overall_sum

    return result





import pandas as pd
df = pd.read_csv("/home/vv2116/1006/cds-bootcamp/homework/nlp/NLP_Final/SummaReranker/src/candidate_generation/candidate_scores_1.csv")
df[df.columns[1]] = df[df.columns[1]].apply(lambda arr : arr.split('|'))
df["r1"] = df["r1"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
df["r2"] = df["r2"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
df["rl"] = df["rl"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
df["scores"] = df.apply(lambda row : [row["r1"],row["r2"],row["rl"]],axis=1)
print(df.shape)
df.head()
# df["scores"][0]



tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
base_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

train_size = 1000
val_size = 100
xsum_train_dataset = MultitaskRerankingDatasetTrain("train", tokenizer, df[df.columns[0]][:train_size].tolist(), df[df.columns[1]][:train_size].tolist(), df[df.columns[2]][:train_size].tolist(),df["scores"][:train_size].tolist(), args.max_len,args.max_summ_len)
xsum_val_dataset = MultitaskRerankingDatasetTrain("val", tokenizer, df[df.columns[0]][train_size:train_size+val_size].tolist(), df[df.columns[1]][train_size:train_size+val_size].tolist(), df[df.columns[2]][train_size:train_size+val_size].tolist(),df["scores"][train_size:train_size+val_size].tolist(), args.max_len,args.max_summ_len)



model = ModelMultitaskBinary(base_model, tokenizer, args)
train_args = TrainingArguments(
    output_dir="models/v5",  # will be changed
    do_train=True,
    do_eval=True,
    do_predict=False,
    num_train_epochs=5,
    optim = "adafactor",
    adafactor=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    # gradient_accumulation_steps = 3,
    max_grad_norm=10e5,
    fp16=True,report_to="wandb",
    logging_dir="logs",
    eval_steps=100,remove_unused_columns=False
)


trainer = CustomTrainer(
    model=model,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    train_dataset=xsum_train_dataset,
    eval_dataset=xsum_val_dataset,
    tokenizer=tokenizer,
    args=train_args
)


# training loop
if True:
    trainer.train()
    model.display_training_labels()