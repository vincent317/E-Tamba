from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from modeling_mamba_transformer import MambaTransformer, ModelArgs, MambaTransformerConfig, MambaTransformerForLM
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

pretrained_mamba_name = 'state-spaces/mamba-1.4b-hf'
pretrained_pythia_name = 'EleutherAI/pythia-1.4b'

tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
seq_len = 1024

#train_file = ["en/c4-train.00000-of-01024.json.gz", "en/c4-train.00001-of-01024.json.gz"]
train_file = ["en/c4-train.00000-of-01024.json.gz"]
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

model = MambaTransformerForLM(
    MambaTransformerConfig(), 
    pretrained_mamba_name=pretrained_mamba_name,
    pretrained_pythia_name=pretrained_pythia_name,
    first_transformer_layers=12,
    mamba_start_layer=36,
    mamba_end_layer=47)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="base_1.4b_1024len_12_12_1_3_epochs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=2200,
    logging_steps=50,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_steps=1100,
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