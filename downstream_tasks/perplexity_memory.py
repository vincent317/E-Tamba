from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from global_code import load_model_tokenizer, device, mamba_hf_name, pythia_name

seq_len = 4096
batch_size = 1
val_file = "en/c4-validation.00000-of-00008.json.gz"
device = 'cuda'
#model_id = '/root/Transformer_Mamba_Transplantation/trans_12_mamba_3_14/checkpoint-17315/model.safetensors'
model_id = pythia_name
with torch.no_grad():
    model, tokenizer = load_model_tokenizer(model_id)
    model.eval()
    breakpoint()
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def prepare_val_dataset(val_file):
        return load_dataset("allenai/c4", data_files=val_file)

    def tokenize(raw_dataset):
        outputs = tokenizer(
            raw_dataset['text'],
            truncation=True,
            max_length=seq_len,
            padding=True,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_mask = []
        
        for input_ids, attn_mask in zip(outputs["input_ids"], outputs["attention_mask"]):
            input_batch.append(input_ids)
            attention_mask.append(attn_mask)
        return {"input_ids": input_batch, "attention_mask": attention_mask}

    raw_data = prepare_val_dataset(val_file)
    tokenized_datasets = raw_data.map(
        tokenize, batched=True, remove_columns=raw_data['train'].column_names
    )
    input_ids = tokenized_datasets['train']['input_ids']
    attn_mask = tokenized_datasets['train']['attention_mask']
    ppxes = []

    for b in tqdm(range(0, len(input_ids), batch_size)):
        batch_input_ids = torch.tensor(input_ids[b:b+batch_size]).to(device)
        batch_attn_mask = torch.tensor(attn_mask[b:b+batch_size]).to(device)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask, labels=batch_input_ids)
        ppx = torch.exp(outputs['loss']).item()
        ppxes.append(ppx)
        print(ppx)

    print(sum(ppxes) / len(ppxes))