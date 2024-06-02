from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_mamba_transformer import MambaTransformerForLM, MambaTransformerConfig
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

seq_len = 256
batch_size = 8
val_file = "en/c4-validation.00000-of-00008.json.gz"
device = 'cuda'

def prepare_val_dataset(val_file):
    return load_dataset("allenai/c4", data_files=val_file)

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

pretrained_mamba_name = 'state-spaces/mamba-1.4b-hf'
pretrained_pythia_name = 'EleutherAI/pythia-1.4b'

tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
raw_data = prepare_val_dataset(val_file)
tokenized_datasets = raw_data.map(
    tokenize, batched=True, remove_columns=raw_data['train'].column_names
)
input_ids = tokenized_datasets['train']['input_ids']
attn_mask = tokenized_datasets['train']['attention_mask']

ppxes = []
checkpoint_point_path = 'sft_1_epoch_100_length_2000_samples/checkpoint-540/model.safetensors'
#model = MambaTransformerForLM(MambaTransformerConfig(), checkpoint_point_path)

model = AutoModelForCausalLM.from_pretrained(pretrained_pythia_name).to(device)
model.eval()

with torch.no_grad():
    for b in tqdm(range(0, len(input_ids), batch_size)):
        batch_input_ids = torch.tensor(input_ids[b:b+batch_size]).to(device)
        batch_attn_mask = torch.tensor(attn_mask[b:b+batch_size]).to(device)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask, labels=batch_input_ids)
        ppx = torch.exp(outputs['loss']).item()
        ppxes.append(ppx)

print(sum(ppxes) / len(ppxes))