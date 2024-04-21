from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.hub import cached_file
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import json
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import pickle 
import gc

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

if __name__ == '__main__':
    pretrained_mamba_name = 'state-spaces/mamba-130m'
    pretrained_pythia_name = 'EleutherAI/pythia-160m'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    seq_len = 256
    train_file = "en/c4-train.00000-of-01024.json.gz"
    val_file = "en/c4-validation.00000-of-00008.json.gz"
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(pretrained_pythia_name).to(device)

    def tokenize(raw_dataset, batch_size=16):
        outputs = tokenizer(
            raw_dataset['text'],
            truncation=True,
            max_length=seq_len,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_mask = []
        probs = []
        
        for length, input_ids, attn_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            if length == seq_len:
                input_batch.append(input_ids)
                attention_mask.append(attn_mask)

        batch_samples = len(input_batch)
        for i in range(0, batch_samples, batch_size):
            input_ids = input_batch[i:min(batch_samples, i+batch_size)]
            attn_masks = attention_mask[i:min(batch_samples, i+batch_size)]
            input_ids = torch.tensor(input_ids).to(device)
            attn_mask = torch.tensor(attn_masks).to(device)
            # Calculate the teacher probabilities
            with torch.no_grad():
                logits = model(input_ids, attention_mask=attn_mask).logits
                probs.extend(list(logits.unbind(dim=0)))
        return {"input_ids": input_batch, "attention_mask": attention_mask, 'probs': probs}
    
    raw_dataset = prepare_dataset(train_file, val_file)
    dataset = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset['train'].column_names
    )

    # 3. Generate new datasets with teacher's probabilities
    file_name = train_file.replace('/', '.') + '_' + val_file.replace('/', '.') +'_dataset.pth'
    torch.save(dataset, file_name)
