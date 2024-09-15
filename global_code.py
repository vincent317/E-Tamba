pythia_name = 'EleutherAI/pythia-1.4b'
mamba_name = 'state-spaces/mamba-1.4b'
mamba_hf_name = 'state-spaces/mamba-1.4b-hf'
recurrent_gemma_name = 'google/recurrentgemma-2b'
device = 'cuda:0'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_mamba_transformer import MambaTransformerForLM, MambaTransformerConfig

def load_model_tokenizer(model_id):
    if 'trans' in model_id:
        model = MambaTransformerForLM(
            MambaTransformerConfig(), 
            pretrained_mamba_name=mamba_name,
            pretrained_pythia_name=pythia_name,
            first_transformer_layers=12,
            mamba_start_layer=25,
            mamba_end_layer=36,
            check_point_path=model_id)
        return model, AutoTokenizer.from_pretrained(pythia_name)
    else:
        return AutoModelForCausalLM.from_pretrained(model_id).to(device), AutoTokenizer.from_pretrained(model_id)