from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from modeling_mamba_transformer import MambaTransformer, ModelArgs, MambaTransformerConfig, MambaTransformerForLM
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

pretrained_mamba_name = 'state-spaces/mamba-130m'
pretrained_pythia_name = 'EleutherAI/pythia-160m'
tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
seq_len = 256
train_file = "en/c4-train.00000-of-01024.json.gz"
val_file = "en/c4-validation.00000-of-00008.json.gz"

def prepare_dataset(train_file, val_file):
    data_files = {"train": train_file, "validation": val_file}
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
        raw_dataset['text'],
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
raw_dataset = prepare_dataset(train_file, val_file)
tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
)

model = MambaTransformerForLM(MambaTransformerConfig(), distilling=True)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="distilling_mamba_everything_unfreezed",
    per_device_train_batch_size=20,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=20000,
    logging_steps=50,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=1.6e-4,
    save_steps=5000,
    fp16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()