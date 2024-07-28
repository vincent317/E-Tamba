from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

def prepare_dataset():
    ds = load_dataset("cimec/lambada", split='test')
    return ds

if __name__ == '__main__':
    dataset = prepare_dataset()
    model_id = 'google/recurrentgemma-2b'
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    correct = 0
    # Tokenize input and prepare for model
    for entry in tqdm(dataset):
        input_ids = tokenizer.encode(entry['text'], return_tensors='pt').to("cuda")
        input_len = len(input_ids[0])

        # Generate model output
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits

        # Get the predicted word
        predicted_index = torch.argmax(predictions[0, -2, :]).item()
        ground_truth = input_ids[:, -1].item()
        if ground_truth == predicted_index:
            correct += 1
    
    print(correct / len(dataset))
