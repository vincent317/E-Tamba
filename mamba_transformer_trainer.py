from transformers import Trainer
import torch
import os


class MambaTransformerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        attn_mask = inputs.pop("attention_mask")
        lm_logits = model(input_ids, attn_mask)

        # labels = input_ids.to(lm_logits.device)
        labels = inputs.pop("labels")
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)