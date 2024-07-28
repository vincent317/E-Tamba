import gzip
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_json_gz(filename, test_size, copy_length):
    with gzip.open(filename, 'r') as f:
        i = 0
        ret = []
        for json_line in f:
            if i == test_size:
                return ret
            data = json.loads(json_line)
            text = data['text']
            if len(text) > copy_length:
                ret.append(text)
                i += 1

def copy_task(test_strings, model=None, tokenizer=None, copy_length=50, shuffle=False):
    success_copies = 0
    for test_string in tqdm(test_strings):
        cur_batch = []
        cur_batch.append(test_string)
        outputs = tokenizer(cur_batch, return_tensors="pt", truncation=True, max_length=copy_length).to(device)
        input_ids = outputs['input_ids']
        attn_masks = outputs['attention_mask']
        if shuffle:
            col_perm = torch.randperm(input_ids.size(1))
            input_ids = input_ids[:, col_perm]
        input_ids = torch.cat([input_ids, input_ids, input_ids], dim=1)
        input_ids = torch.cat([input_ids, input_ids[:, 0:1]], dim=1)
        output_ids = model.generate(input_ids, attention_mask = attn_masks, max_new_tokens = copy_length-1)
        gold_token_len = (input_ids.shape[1]-1) // 2
        if torch.equal(input_ids[0][:gold_token_len], output_ids[0][gold_token_len*2:]):
            success_copies += 1
    return success_copies / len(test_strings)

if __name__ == "main":
    model_id = 'EleutherAI/pythia-1.4b'
    test_file = 'c4-validation.00000-of-00008.json.gz'
    copy_length = 50
    test_size = 10
    test_strings = load_json_gz(test_file, test_size, copy_length)
    device = 'cuda:0'

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(copy_task(test_strings, model, tokenizer, copy_length, False))
