from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from modeling_mamba_transformer import MambaTransformer, ModelArgs
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import gzip

num_rows = 1024

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]

def load_json_gz(filename, num_inputs):
  with gzip.open(filename, 'r') as f:
      i = 0
      ret = []
      for json_line in f:
          if i == num_inputs:
              return ret
          data = json.loads(json_line)
          ret.append(data['text'])
          i += 1
data = load_json_gz('c4-train.00000-of-01024.json.gz', num_rows) # 356317 lines

device = 'cuda'
pretrained_mamba_name = 'state-spaces/mamba-130m'
pretrained_pythia_name = 'EleutherAI/pythia-160m'
tokenizer = AutoTokenizer.from_pretrained(pretrained_pythia_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

model = MambaTransformer.from_pretrained(pretrained_mamba_name, pretrained_pythia_name)
# model = AutoModelForCausalLM.from_pretrained(pretrained_pythia_name).to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
max_seq_len = 512

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer(data, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
losses = []

batch_size = 8
num_tokens_per_row = input_ids.size()[1]

for i in tqdm(range(0, num_rows, batch_size)):
    mask = attention_mask[i:i+batch_size, :num_tokens_per_row-1].to(device)
    inputs = input_ids[i:i+batch_size, :num_tokens_per_row-1].to(device)
    targets = input_ids[i:i+batch_size, 1:num_tokens_per_row].to(device)

    #outputs = model(input_ids=inputs, attention_mask=mask).logits # use this for AutoModelForCasualLM models
    outputs = model(input_ids=inputs, attention_mask=mask) # use this for homemade MambaTransformer models
    
    loss = torch.exp(criterion(outputs.transpose(1, 2), targets)).item()
    losses.append(loss) # perplexity

print(sum(losses) / len(losses))