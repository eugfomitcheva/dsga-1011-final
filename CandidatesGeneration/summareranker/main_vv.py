from datasets import load_dataset
from datasets import get_dataset_split_names
from model import ModelMultitaskBinary
import argparse
import torch
import torch.nn as nn
from model import ModelMultitaskBinary
from training_utils import *
# from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel,BertTokenizer, BertTokenizerFast, BertModel
from transformers import PegasusForConditionalGeneration,PegasusTokenizer
import pandas as pd
from dataset import *


parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
parser.add_argument('--expert_hidden_size', type=int, default=1024)
parser.add_argument('--tower_hidden_size', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=1024) # 768 / 1024
parser.add_argument('--bottom_hidden_size', type=int, default=1024)
parser.add_argument('--num_experts', type=int, default=6)
parser.add_argument('--scoring_methods', type=str, default = ["rouge_1","rouge_l"])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--max_summ_len', type=int, default=64)


args = parser.parse_args("")
args.n_tasks = len(args.scoring_methods)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args.device = device

xsum_dataset = load_dataset("xsum")


tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
base_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

model = ModelMultitaskBinary(base_model, tokenizer, args)

df = pd.read_csv("candidate_scores_samples.csv")
df["r1"] = df["r1"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
df["r2"] = df["r2"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
df["rls"] = df["rls"].apply(lambda arr : [float(val) for val in arr[2:-2].split(',')])
# df["rls"][1][0]

df.head()
dataset = MultitaskRerankingDatasetTrain("train", tokenizer, df[df.columns[0]].tolist(), df[df.columns[1]].tolist(), df[df.columns[2]].tolist(),df["rls"].tolist(), args.max_len,args.max_summ_len)

from transformers import Trainer, TrainingArguments, default_data_collator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader

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
            shuffle=train_dataset.args.shuffle_train,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
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


args.max_train_size = 1000
args.max_val_size=100

train_dataset = dataset
train_dataset.texts = dataset.texts[:args.max_train_size]
train_dataset.summaries = dataset.summaries[:args.max_train_size]
train_dataset.labels = dataset.labels[:args.max_train_size]
train_dataset.scores = dataset.scores[:args.max_train_size]


val_dataset = dataset
val_dataset.texts = dataset.texts[args.max_train_size:args.max_train_size+args.max_val_size]
val_dataset.summaries = dataset.summaries[args.max_train_size:args.max_train_size+args.max_val_size]
val_dataset.labels = dataset.labels[args.max_train_size:args.max_train_size+args.max_val_size]
val_dataset.scores = dataset.scores[args.max_train_size:args.max_train_size+args.max_val_size]

trainer = CustomTrainer(
    model=model,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

if True:
    results = trainer.evaluate()
    print("*" * 50, "Init VAL results:")
    print(results)
    model.moe.display_tasks_probs()

# training loop
if True:
    trainer.train()
    model.display_training_labels()
