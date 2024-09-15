import gzip
import os
import sys
import json
import torch
import random
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from global_code import load_model_tokenizer, device, mamba_hf_name

def load_data(filename):
    ret = []
    with gzip.open(filename, 'r') as f:
        for json_line in f:
            ret.append(json_line)
    return ret

def select_data(arr, test_size, copy_length):
    random.shuffle(arr)
    ret = []
    i = 0
    for line in arr:
        if i == test_size:
            return ret
        data = json.loads(line)
        text = data['text']
        if len(text.split(' ')) > copy_length:
            ret.append(text)
            i += 1
    return None

def copy_task(test_strings, model=None, tokenizer=None, copy_length=50, copy_times = 2, shuffle=False):
    success_copies = 0
    single_one = torch.ones((1, 1)).to(device)
    for test_string in tqdm(test_strings):
        cur_batch = []
        cur_batch.append(test_string)
        outputs = tokenizer(cur_batch, return_tensors="pt", truncation=True, max_length=copy_length).to(device)
        input_ids = outputs['input_ids']
        attn_mask = outputs['attention_mask']
        if shuffle:
            col_perm = torch.randperm(input_ids.size(1))
            input_ids = input_ids[:, col_perm]
        input_ids_old = input_ids
        attn_mask_old = attn_mask
        for _ in range(copy_times-1):
            input_ids = torch.cat((input_ids, input_ids_old), dim=1)
            attn_mask = torch.cat((attn_mask, attn_mask_old), dim=1)
        input_ids = torch.cat((input_ids, input_ids[:, 0:2]), dim=1)
        attn_mask = torch.cat((attn_mask, single_one, single_one), dim=1)
        gold_token_len = (input_ids.shape[1]-2) // copy_times

        generated_ids = input_ids
        needed_tokens = input_ids_old.size()[1]-2
        for _ in range(needed_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated_ids, attention_mask=attn_mask)
            try:
                logits = outputs.logits[:, -1, :]
            except:
                logits = outputs['logits'][:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            attn_mask = torch.cat([attn_mask, single_one], dim=1)
        if torch.equal(input_ids[0][:gold_token_len], generated_ids[0][gold_token_len*copy_times:]):
            success_copies += 1
    return success_copies / len(test_strings)
    
if __name__ == "__main__":
   # model_id = '/root/Transformer_Mamba_Transplantation/trans_12_mamba_3_14/checkpoint-17315/model.safetensors'
    model_id = mamba_hf_name
    test_file = './c4-validation.00000-of-00008.json.gz'
    dataset = load_data(test_file)
    
    copy_times = 2
    test_size = 30
    # test_copy_lengths = [50, 100, 200, 400]
    test_copy_lengths = [50, 100, 200, 400]

    model, tokenizer = load_model_tokenizer(model_id)
    for copy_length in test_copy_lengths:
    
        test_strings = select_data(dataset, test_size, copy_length)
        res = copy_task(test_strings, model, tokenizer, copy_length, copy_times, True)
        print(
            f"\nModel: {model_id}\n"
            f"Copy Times: {copy_times}\n"
            f"Copy Length: {copy_length}\n"
            f"Test Size: {test_size}\n"
            f"Accuracy: {res}\n\n"
        )