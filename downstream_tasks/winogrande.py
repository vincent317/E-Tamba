import torch
import os
import sys
from tqdm import tqdm
from datasets import load_dataset
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from global_code import load_model_tokenizer, device, pythia_name, mamba_hf_name, recurrent_gemma_name

# Following GPT-3's implementation: https://arxiv.org/pdf/2005.14165

def prepare_input(context, option):
    return context.replace('_', option)

def evaluate_example(model, tokenizer, context, option1, option2):
    # Prepare the input texts
    input_text_1 = prepare_input(context, option1)
    input_text_2 = prepare_input(context, option2)
    
    # Tokenize the inputs
    inputs_1 = tokenizer(input_text_1, return_tensors="pt").to(device)
    inputs_2 = tokenizer(input_text_2, return_tensors="pt").to(device)

    # Get the lengths of the input sequences
    len_1 = inputs_1.input_ids.size(1)
    len_2 = inputs_2.input_ids.size(1)
    # Pad the shorter sequence
    if len_1 > len_2:
        padding = torch.full((1, len_1 - len_2), tokenizer.eos_token_id, dtype=torch.long).to(device)
        inputs_2_padded = torch.cat([inputs_2.input_ids, padding], dim=1)
        inputs_1_padded = inputs_1.input_ids
    elif len_2 > len_1:
        padding = torch.full((1, len_2 - len_1), tokenizer.eos_token_id, dtype=torch.long).to(device)
        inputs_1_padded = torch.cat([inputs_1.input_ids, padding], dim=1)
        inputs_2_padded = inputs_2.input_ids
    else:
        inputs_1_padded = inputs_1.input_ids
        inputs_2_padded = inputs_2.input_ids
    # Find the indices where the two sequences differ
    diff_index = (inputs_1_padded[0] != inputs_2_padded[0]).nonzero(as_tuple=True)[0][0]

    # Get the model's output logits
    with torch.no_grad():
        input_ids_1, attn_mask_1 = inputs_1['input_ids'], inputs_1['attention_mask']
        input_ids_2, attn_mask_2 = inputs_2['input_ids'], inputs_2['attention_mask']
        outputs1 = model(input_ids=input_ids_1, attention_mask=attn_mask_1)
        outputs2 = model(input_ids=input_ids_2, attention_mask=attn_mask_2)
        try:
            logits_1 = outputs1.logits
            logits_2 = outputs2.logits
        except:
            logits_1 = outputs1['logits']
            logits_2 = outputs2['logits']
    
     # Slice the logits to get the part after the option token
    logits_1_after_option = logits_1[:, diff_index:-1, :]  # Assuming you're ignoring the last [SEP] token
    logits_2_after_option = logits_2[:, diff_index:-1, :]

    # Get the target tokens (shifted by one)
    target_tokens_1 = inputs_1.input_ids[:, diff_index + 1:]
    target_tokens_2 = inputs_2.input_ids[:, diff_index + 1:]

    # Calculate loss using CrossEntropyLoss
    loss_fct = torch.nn.CrossEntropyLoss()

    loss_1 = loss_fct(logits_1_after_option.view(-1, logits_1_after_option.size(-1)),
                      target_tokens_1.view(-1))
    loss_2 = loss_fct(logits_2_after_option.view(-1, logits_2_after_option.size(-1)),
                      target_tokens_2.view(-1))
    # Determine which option the model prefers
    return 1 if loss_1 < loss_2 else 2

if __name__ == '__main__':
    # Load the WinoGrande dataset
    dataset = load_dataset("winogrande", "winogrande_xl")  # or another size like "winogrande_l"

    # Replace with your model's name
    model_id = recurrent_gemma_name
    #model_id = '/root/Transformer_Mamba_Transplantation/trans_12_mamba_3_14/checkpoint-17315/model.safetensors'
    model, tokenizer = load_model_tokenizer(model_id)
    correct = 0
    total = 0

    for example in dataset['validation']:
        context = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = example['answer']
        
        prediction = evaluate_example(model, tokenizer, context, option1, option2)
        print(f"Context: {context}\nOption 1: {option1}\nOption 2: {option2}\nPrediction: {prediction}\nAnswer: {answer}\n")
        if prediction == int(answer):
            correct += 1
        total += 1
        # except Exception as e:
        #     print(f"Error {e} Context: {context}\nOption 1: {option1}\nOption 2: {option2}\nAnswer: {answer}\n")
        #     continue

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")