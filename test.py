from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from modeling_mamba_transformer import MambaTransformer, ModelArgs
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]

device = 'cuda'
pretrained_mamba_name = 'state-spaces/mamba-130m'
pretrained_pythia_name = 'EleutherAI/pythia-160m'
tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name)
model = MambaTransformer.from_pretrained(pretrained_mamba_name, pretrained_pythia_name)
#model = AutoModelForCausalLM.from_pretrained(pretrained_pythia_name).to(device)
criterion = nn.CrossEntropyLoss()

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
ids = encodings['input_ids']
seq_len = 1024
losses = []

for i in tqdm(range(1, ids.size(1) - seq_len, seq_len)):
    inputs = ids[:, i:i+seq_len].to(device)
    targets = ids[:, (i-1):(i-1)+seq_len].to(device)

    
    #outputs = model(inputs).logits # use this for AutoModelForCasualLM models
    outputs = model(inputs) # use this for homemade MambaTransformer models

    loss = criterion(outputs.transpose(1,2), targets)
    losses.append(loss.item()) # cross-entropy loss

print(sum(losses) / len(losses))