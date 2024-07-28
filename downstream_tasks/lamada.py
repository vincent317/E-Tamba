from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

def prepare_dataset():
    ds = load_dataset("cimec/lambada", split='test')
    return ds

if __name__=='main':
    dataset = prepare_dataset()
    breakpoint()
    model_id = 'EleutherAI/pythia-1.4b'
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize input and prepare for model
    for entry in ds:
        input_ids = tokenizer.encode(entry, return_tensors='pt')
        input_len = len(input_ids[0])

        # Generate model output
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits

        # Get the predicted word
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_word = tokenizer.decode([predicted_index])