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
train_file = ["en/c4-train.00002-of-01024.json.gz"]
val_file = "en/c4-validation.00000-of-00008.json.gz"

def prepare_dataset(train_file, val_file):
    data_files = {"train": train_file, "validation": val_file}
    c4_train_subset = load_dataset("allenai/c4", data_files=data_files, split='train')
    c4_valid_subset = load_dataset("allenai/c4", data_files=data_files, split='validation')
    return DatasetDict(
        {
            "train": c4_train_subset.shuffle().select(range(15000)),
            "valid": c4_valid_subset.shuffle().select(range(1000))
        }
    )

def tokenize(raw_dataset):
    base_seq_len = 50
    outputs = tokenizer(
        raw_dataset['text'],
        truncation=True,
        max_length=base_seq_len,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    attention_mask = []

    for length, input_ids, attn_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
        if length == base_seq_len:
            input_ids = input_ids + input_ids + input_ids
            attn_mask = attn_mask + attn_mask + attn_mask
            input_batch.append(input_ids)
            attention_mask.append(attn_mask)
    return {"input_ids": input_batch, "attention_mask": attention_mask}

if __name__ == '__main__':
    raw_dataset = prepare_dataset(train_file, val_file)
    tokenized_datasets = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
    )

    model = MambaTransformerForLM(MambaTransformerConfig(), 
                                check_point_path='seq_len_3_epochs_6_6_1/checkpoint-10000/model.safetensors',
                                sft=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="seq_len_3_epochs_1024_6_6_1_2_train_set_sft",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=800,
        logging_steps=50,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_steps=800,
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