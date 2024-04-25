from __future__ import annotations
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from transformers import GPTNeoXLayer, GPTNeoXConfig, PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import create_block
from safetensors.torch import load_file
from transformers.debug_utils import detect_overflow

@dataclass
class ModelArgs:
    d_model: int
    first_transformer_layers: int
    mamba_layers: int
    vocab_size: int
    transformer_config: GPTNeoXConfig
    mamba_config: dict

class MambaTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.max_len = 1024
        self.embed_in = nn.Embedding(args.vocab_size, args.d_model)
        self.emb_dropout = nn.Dropout(args.transformer_config.hidden_dropout)
        self.first_transformer_layers = nn.ModuleList([GPTNeoXLayer(args.transformer_config) for _ in range(args.first_transformer_layers)])

        self.mamba_layers = nn.ModuleList(
            [
                create_block(
                    d_model=args.d_model,
                    rms_norm=False,
                    residual_in_fp32=True,
                    fused_add_norm=False,
                    layer_idx=i,
                )
                for i in range(args.mamba_layers)
            ]
        )
        self._use_flash_attention_2 = args.transformer_config._attn_implementation == "flash_attention_2"
        self.final_transformer_layer = GPTNeoXLayer(args.transformer_config)
        self.final_layer_norm = nn.LayerNorm(args.transformer_config.hidden_size, eps=args.transformer_config.layer_norm_eps)
        self.embed_out = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.embed_out.weight = self.embed_in.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper
        # self.connection_layer = nn.Linear(args.transformer_config.hidden_size, args.transformer_config.hidden_size)
        # torch.nn.init.eye_(self.connection_layer.weight)
        # torch.nn.init.zeros_(self.connection_layer.bias)
    
    def forward(self, input_ids, attention_mask, attn_dtype):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if 0 in attention_mask else None
        else:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(attn_dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_dtype).min
        past_length = 0
        device = input_ids.device
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        x = self.embed_in(input_ids)
        x = self.emb_dropout(x)
        
        head_mask = [None] * (self.args.first_transformer_layers+1)
        for i, layer in enumerate(self.first_transformer_layers):
            outputs = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=True,
            )
            x = outputs[0]

        with torch.cuda.amp.autocast(enabled=False):
            # x = self.connection_layer(x)
            residual = None
            for layer in self.mamba_layers:
                x, residual = layer(
                    x, residual
                )
            x = x + residual
            x = self.final_transformer_layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[self.args.first_transformer_layers],
                use_cache=True,
            )[0]

            x = self.final_layer_norm(x)
            logits = self.embed_out(x)
            return logits
    
    @staticmethod
    def from_pretrained(pretrained_mamba_name: str, pretrained_pythia_name: str, first_transformer_layers=7, mamba_start_layer=20, mamba_end_layer=23):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        mamba_config_data = load_config_hf(pretrained_mamba_name)
        pythia_config_data = GPTNeoXConfig.from_pretrained(pretrained_pythia_name)

        # Originally we have 12 transformer layers, now we keep 8 and replace the next 3 with 4 mamba layers. 
        # But we still keep the last transformer layer.
        args = ModelArgs(
            d_model=mamba_config_data['d_model'],
            mamba_layers=mamba_end_layer-mamba_start_layer+1,
            first_transformer_layers=first_transformer_layers,
            vocab_size=pythia_config_data.vocab_size,
            transformer_config=pythia_config_data,
            mamba_config=mamba_config_data
        )
        model = MambaTransformer(args).to('cuda')
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cuda', mmap=True)
        
        mamba_state_dict, pythia_state_dict = load_state_dict_hf(pretrained_mamba_name), load_state_dict_hf(pretrained_pythia_name)

        transformer_target_layers_set = set()
        mamba_target_layers_set = set()

        for i in range(first_transformer_layers):
            transformer_target_layers_set.add(i)
        transformer_target_layers_set.add(pythia_config_data.num_hidden_layers-1)
        for i in range(mamba_start_layer, mamba_end_layer+1):
            mamba_target_layers_set.add(i)
        
        pattern = r".layers\.(\d+)\."

        new_state_dict = {}
        for key in mamba_state_dict:
            match = re.search(pattern, key)
            if match:
                layer_index = int(match.group(1))
                if layer_index in mamba_target_layers_set:
                    new_key = key.replace('backbone.layers', 'mamba_layers').replace(str(layer_index), str(layer_index-mamba_start_layer))
                    new_state_dict[new_key] = mamba_state_dict[key]

        for key in pythia_state_dict:
            if 'embed' in key or 'final_layer_norm' in key:
                new_key = key.replace('gpt_neox.', '')
                new_state_dict[new_key] = pythia_state_dict[key]
            else:
                match = re.search(pattern, key)
                if match:
                    layer_index = int(match.group(1))
                    if layer_index in transformer_target_layers_set:
                        if layer_index == pythia_config_data.num_hidden_layers-1:
                            new_key = key.replace('gpt_neox.layers.', '').replace(str(layer_index), 'final_transformer_layer')
                        else:
                            new_key = key.replace('gpt_neox.layers', 'first_transformer_layers')
                        new_state_dict[new_key] = pythia_state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)
        model.embed_in = nn.Embedding.from_pretrained(pythia_state_dict['gpt_neox.embed_in.weight'].to(model.embed_in.weight.dtype))
        model.embed_in.requires_grad_ = False
        return model
    
    def freeze_layers_except_mamba(self):
        """Freezes all parameters except for those in the Mamba layers."""
        # Freeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze parameters in the Mamba layers and projection layers
        # for param in self.connection_layer.parameters():
        #     param.requires_grad = True
        for layer in self.mamba_layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.final_transformer_layer.parameters():
            param.requires_grad = True
        for param in self.final_layer_norm.parameters():
            param.requires_grad = True
        for param in self.embed_out.parameters():
            param.requires_grad = True


class MambaTransformerForLM(PreTrainedModel):
    def __init__(self, 
            config=None, 
            check_point_path=None, 
            distilling=False, 
            T=4, 
            distill_loss_weight=0.5, 
            first_transformer_layers=7, 
            mamba_start_layer=20, 
            mamba_end_layer=23):
        super().__init__(config)
        pretrained_mamba_name = 'state-spaces/mamba-130m'
        pretrained_pythia_name = 'EleutherAI/pythia-160m'
        self.model = MambaTransformer.from_pretrained(pretrained_mamba_name, 
                                                      pretrained_pythia_name, 
                                                      first_transformer_layers, 
                                                      mamba_start_layer, 
                                                      mamba_end_layer)
        if check_point_path is not None:
            loaded = load_file(check_point_path)
            keys_to_change = list(loaded.keys())  # Create a list of keys to iterate over
            for key in keys_to_change:
                new_key = key.replace('model.', '')
                loaded[new_key] = loaded.pop(key)  # Move the value to the new key and remove the old key
            self.model.load_state_dict(loaded, strict=False)
        self.model.freeze_layers_except_mamba()
        self.teacher = None
        if distilling:
            self.batch_count = 0
            device = 'cuda'
            self.teacher = AutoModelForCausalLM.from_pretrained(pretrained_pythia_name).to(device)
            self.T = T
            self.distill_loss_weight = distill_loss_weight
            self.log_steps = 0
            self.ce_loss_sum = 0
            self.distill_loss_sum = 0

    def forward(self, input_ids, attention_mask, labels):
        logits = self.model(input_ids, attention_mask, self.dtype)
        
        if labels is None:
            return {"logits": logits}
        else:
            cross_entropy_fcn = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            cross_entropy_loss = cross_entropy_fcn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            
            if self.teacher is not None:
                kl_loss = nn.KLDivLoss(reduction="batchmean")
                self.batch_count += 1
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        teacher_logits = self.teacher(input_ids, attention_mask=attention_mask).logits
                s_log_probs = F.log_softmax(logits/self.T, dim=-1)
                t_probs = F.softmax(teacher_logits/self.T, dim=-1)
                distill_loss = kl_loss(s_log_probs, t_probs) / t_probs.size()[1] * (self.T**2)
                total_loss = self.distill_loss_weight*distill_loss + (1-self.distill_loss_weight)*cross_entropy_loss
                self.ce_loss_sum += cross_entropy_loss.item()
                self.distill_loss_sum += distill_loss.item()

                if self.batch_count == 100:
                    self.log_steps += 50
                    s = "Step:" + str(self.log_steps) + ",CE loss:" + str(self.ce_loss_sum / 100) + ",Soft loss:" + str(self.distill_loss_sum / 100)+'\n'
                    print(s)
                    # Open a file in append mode
                    with open('record.txt', 'a') as file:
                        file.write(s)
                    self.ce_loss_sum = 0
                    self.distill_loss_sum = 0
                    self.batch_count = 0

                return {"loss": total_loss, "logits": logits}        
            else:
                return {"loss": cross_entropy_loss, "logits": logits}        
    
class MambaTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)