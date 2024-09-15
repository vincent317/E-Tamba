import os
import sys
from datasets import load_dataset
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from global_code import load_model_tokenizer, device, pythia_name, mamba_hf_name, recurrent_gemma_name
import torch
from tqdm import tqdm

def prepare_dataset():
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    return ds

if __name__ == '__main__':
    dataset = prepare_dataset()
    model_id = '/root/Transformer_Mamba_Transplantation/trans_12_mamba_3_14/checkpoint-17315/model.safetensors'
    #model_id = recurrent_gemma_name
    model, tokenizer = load_model_tokenizer(model_id)
    correct = 0
    # Tokenize input and prepare for model
    for entry in tqdm(dataset):
        tokenized = tokenizer(entry['text'], return_tensors='pt').to(device)
        input_ids = tokenized['input_ids']
        attn_mask = tokenized['attention_mask']

        # Generate model output
        with torch.no_grad():
            outputs = model(input_ids=input_ids[:, :-1], attention_mask=attn_mask[:, :-1])
            try:
                predictions = outputs.logits
            except:
                predictions = outputs['logits']

        # Get the predicted word
        predicted_index = torch.argmax(predictions[0, -1, :], dim=-1).item()
        ground_truth = input_ids[:, -1].item()
        if ground_truth == predicted_index:
            correct += 1
    
    print(correct / len(dataset))
