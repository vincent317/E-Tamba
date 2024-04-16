from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from modeling_mamba_transformer import MambaTransformer, ModelArgs
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import gzip
from datasets import load_dataset, Dataset, DatasetDict
from mamba_transformer_trainer import MambaTransformerTrainer


pretrained_mamba_name = 'state-spaces/mamba-130m'
pretrained_pythia_name = 'EleutherAI/pythia-160m'
tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name)
seq_len = 128

def prepare_dataset():
    data_files = {"train": "en/c4-train.00000-of-01024.json.gz", "validation": "en/c4-validation.00000-of-00008.json.gz"}
    c4_train_subset = load_dataset("allenai/c4", data_files=data_files, split='train')
    c4_valid_subset = load_dataset("allenai/c4", data_files=data_files, split='validation')
    return DatasetDict(
        {
            "train": c4_train_subset,  # .shuffle().select(range(50000)),
            "valid": c4_valid_subset,  # .shuffle().select(range(500))
        }
    )

def tokenize(raw_dataset):
    outputs = tokenizer(
        raw_dataset['text'][:10],
        truncation=True,
        max_length=seq_len,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    attention_mask = []
    for length, input_ids, attn_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
        if length == seq_len:
            input_batch.append(input_ids)
            attention_mask.append(attn_mask)
    return {"input_ids": input_batch, "attention_mask": attention_mask}

raw_dataset = prepare_dataset()
tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
)

model = MambaTransformer.from_pretrained(pretrained_mamba_name, pretrained_pythia_name)
model.freeze_layers_except_mamba()
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="mamba_transformer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=5_00,
    logging_steps=5_0,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-5,
    save_steps=5_00,
    fp16=True,
)

trainer = MambaTransformerTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()