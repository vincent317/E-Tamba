from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from modeling_mamba_transformer import MambaTransformer, ModelArgs, MambaTransformerConfig, MambaTransformerForLM
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

pretrained_mamba_name = 'state-spaces/mamba-1.4b'
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
            "train": c4_train_subset.shuffle().select(range(100000)),
            "valid": c4_valid_subset.shuffle().select(range(10000))
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
        #if length == seq_len:
        input_batch.append(input_ids)
        attention_mask.append(attn_mask)
    return {"input_ids": input_batch, "attention_mask": attention_mask}
raw_dataset = prepare_dataset(train_file, val_file)
tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# model = MambaTransformerForLM(
#     MambaTransformerConfig(), 
#     pretrained_mamba_name=pretrained_mamba_name,
#     pretrained_pythia_name=pretrained_pythia_name,
#     first_transformer_layers=12,
#     mamba_start_layer=36,
#     mamba_end_layer=47)

# args = TrainingArguments(
#     output_dir="trans_12_mamba_36_47",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     eval_strategy="steps",
#     eval_steps=1000,
#     logging_steps=10,
#     gradient_accumulation_steps=16,
#     num_train_epochs=1,
#     learning_rate=1e-4,
#     lr_scheduler_type='constant',
#     save_steps=1000,
#     max_grad_norm=3,
#     bf16=True,
#     warmup_ratio=0.03
# )

# model = MambaTransformerForLM(
#     MambaTransformerConfig(), 
#     pretrained_mamba_name=pretrained_mamba_name,
#     pretrained_pythia_name=pretrained_pythia_name,
#     first_transformer_layers=12,
#     mamba_start_layer=25,
#     mamba_end_layer=36)

# args = TrainingArguments(
#     output_dir="trans_12_mamba_25_36",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     eval_strategy="steps",
#     eval_steps=900,
#     logging_steps=10,
#     gradient_accumulation_steps=16,
#     num_train_epochs=1,
#     learning_rate=1e-4,
#     lr_scheduler_type='constant',
#     save_steps=1800,
#     max_grad_norm=3,
#     bf16=True,
#     warmup_ratio=0.03
# )

# model = MambaTransformerForLM(
#     MambaTransformerConfig(), 
#     pretrained_mamba_name=pretrained_mamba_name,
#     pretrained_pythia_name=pretrained_pythia_name,
#     first_transformer_layers=12,
#     mamba_start_layer=14,
#     mamba_end_layer=25)

# args = TrainingArguments(
#     output_dir="trans_12_mamba_14_25",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     eval_strategy="steps",
#     eval_steps=900,
#     logging_steps=10,
#     gradient_accumulation_steps=16,
#     num_train_epochs=1,
#     learning_rate=1e-4,
#     lr_scheduler_type='constant',
#     save_steps=1800,
#     max_grad_norm=3,
#     bf16=True,
#     warmup_ratio=0.03
# )

model = MambaTransformerForLM(
    MambaTransformerConfig(), 
    pretrained_mamba_name=pretrained_mamba_name,
    pretrained_pythia_name=pretrained_pythia_name,
    first_transformer_layers=12,
    mamba_start_layer=3,
    mamba_end_layer=14)

args = TrainingArguments(
    output_dir="trans_12_mamba_3_14",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=900,
    logging_steps=10,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=1e-4,
    lr_scheduler_type='constant',
    save_steps=1800,
    max_grad_norm=3,
    bf16=True,
    warmup_ratio=0.03
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