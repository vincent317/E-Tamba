import ast
import random
import torch
import os
import sys
import torch.nn as nn
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from global_code import load_model_tokenizer, device, mamba_hf_name, pythia_name, recurrent_gemma_name

def prepare_phonebook():
    name_phone_pairs = []
    with open('./phonebook.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line[-1] == ',':
                line = line[:-1]
            pair = ast.literal_eval(line)
            name_phone_pairs.append((pair[0], pair[1]))
    return name_phone_pairs

def phone_book_task(phonebook, test_size=10, max_book_size=20, hint_chars=2, model=None, tokenizer=None):
    single_one = torch.ones((1, 1)).to(device)
    book = ''
    success_lookups = 0
    for i in range(max_book_size):
        name = phonebook[i][0]
        phone = phonebook[i][1]
        book = book + name + ': ' + phone + '\n'
    with torch.no_grad():
        for _ in tqdm(range(test_size)):
            max_num_tokens = 15
            chosen_idx = random.randint(0, max_book_size)
            query = book + phonebook[chosen_idx][0] + ': ' + phonebook[chosen_idx][1][:hint_chars]
            gt_number = phonebook[chosen_idx][1]
            inputs = tokenizer(query, return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]

            for _ in range(max_num_tokens-1):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                try:
                    logits = outputs.logits[:, -1, :]
                except:
                    logits = outputs['logits'][:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                attn_mask = torch.cat((attn_mask, single_one), dim=1)
            output_answer = tokenizer.decode(input_ids[0])
            if output_answer.count(gt_number) > 1:
                success_lookups += 1
    return success_lookups / test_size

if __name__ == '__main__':
    #model_id = mamba_hf_name
    model_id = '/root/Transformer_Mamba_Transplantation/trans_12_mamba_3_14/checkpoint-17315/model.safetensors'
    phonebook = prepare_phonebook()
    test_size = 30
    hint_chars = 3
    max_book_sizes = [20, 40, 100]
    model, tokenizer = load_model_tokenizer(model_id)
    for max_book_size in max_book_sizes:
        acc = phone_book_task(phonebook, test_size, max_book_size, hint_chars, model, tokenizer)
        print(
            f"\nModel: {model_id}\n"
            f"Book Size: {max_book_size}\n"
            f"Test Size: {test_size}\n"
            f"Accuracy: {acc}\n\n"
        )