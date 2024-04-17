from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import numpy as np
import json
import gzip
import json
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import pickle 


pretrained_mamba_name = 'state-spaces/mamba-130m'
pretrained_pythia_name = 'EleutherAI/pythia-160m'
tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name)
seq_len = 128

def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                        _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

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
    
# Angular distance from: https://arxiv.org/pdf/2403.17887.pdf
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

device = 'cuda'
model = AutoModel.from_pretrained(pretrained_pythia_name, output_hidden_states=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
  pretrained_pythia_name,
  padding_side="left",
  cache_dir=pretrained_pythia_name+'_tokenizer',
)
tokenizer.pad_token = tokenizer.eos_token




# Function to get second last hidden states
def get_second_last_hidden_states(model, tokenized_batch, device='cuda'):
    # Move input_ids and attention_mask to the specified device
    input_ids = torch.tensor(tokenized_batch['input_ids']).to(device)
    # print(input_ids)
    input_ids = input_ids.unsqueeze(0)

    
    attention_mask = torch.tensor(tokenized_batch['attention_mask']).to(device)

    attention_mask = attention_mask.unsqueeze(0)

    # Forward pass, ensure you set output_hidden_states=True when loading the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # outputs.hidden_states is a tuple where each element corresponds to the output
    # from a specific layer. For Pythia, the second last layer would be -2.
    second_last_hidden_states = outputs.hidden_states[-2]

    return second_last_hidden_states

all_second_last_hidden_states = []

# Assuming `tokenized_datasets` is a DatasetDict containing 'train' and 'valid'
for batch in tqdm(tokenized_datasets['train']):
    # Get second last hidden states for the current batch
    second_last_hidden_states = get_second_last_hidden_states(model, batch)
    
    # Move the hidden states back to CPU memory and convert to numpy for storage
    # This step is crucial to avoid filling up your GPU memory
    second_last_hidden_states = second_last_hidden_states.cpu().numpy()
    
    # Append the current batch's hidden states to the list
    all_second_last_hidden_states.append(second_last_hidden_states)

all_second_last_hidden_states = np.concatenate(all_second_last_hidden_states, axis=0)
pickle.dump(all_second_last_hidden_states, open("hidden_state", "wb"))

print("Collected second last hidden states shape:", all_second_last_hidden_states.shape)
