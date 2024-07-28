import ast
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def phone_book_task(phonebook, test_size=10, max_book_size=20, model=None, tokenizer=None):
    softmax = nn.Softmax(dim=2)
    book = ''
    success_lookups = 0
    for i in range(max_book_size):
        name = phonebook[i][0]
        phone = phonebook[i][1]
        book = book + name + ': ' + phone + '.\n'
    book += 'Liam: 436-725-2906\nOlivia: 192-311-5790\n\n'
    with torch.no_grad():
        for _ in tqdm(range(test_size)):
            max_num_tokens = 30
            chosen_idx = random.randint(2, max_book_size)
            query = book + phonebook[chosen_idx][0] + ':'
            gt_number = phonebook[chosen_idx][1]
            input_ids = tokenizer(query, return_tensors="pt", padding=True).to("cuda")["input_ids"]

            for i in range(max_num_tokens-1):
                bs, seq_len = input_ids.size()
                mask = torch.ones(bs, seq_len).to('cuda')
                logits = model(input_ids=input_ids, attention_mask=mask, labels=None)['logits'] # bs, seq_len, vocab_size
                next_token = torch.unsqueeze(torch.argmax(softmax(logits), dim=-1)[:, -1], 1)
                input_ids = torch.cat((input_ids, next_token), dim=-1) # bs, seq_len, 1
            
            output_answer = tokenizer.decode(input_ids[0])
            if output_answer.count(gt_number) > 1:
                success_lookups += 1
    return success_lookups / test_size

if __name__ == 'main':
    model_id = 'EleutherAI/pythia-1.4b'
    phonebook = prepare_phonebook()
    test_size = 10
    max_book_size = 20

    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(phone_book_task(phonebook, test_size, max_book_size, model, tokenizer))