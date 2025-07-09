import torch
from transformers import Trainer


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device

        chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"].to(device),
            attention_mask=inputs["attention_mask_chosen"].to(device)
        )
        rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"].to(device),
            attention_mask=inputs["attention_mask_rejected"].to(device)
        )

        chosen_reward = chosen_outputs.logits.squeeze(-1)
        rejected_reward = rejected_outputs.logits.squeeze(-1)

        # Ensure dimensions
        if chosen_reward.dim() == 0:
            chosen_reward = chosen_reward.unsqueeze(0)
        if rejected_reward.dim() == 0:
            rejected_reward = rejected_reward.unsqueeze(0)

        loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()

        if return_outputs:
            return loss, {
                "chosen_reward": chosen_reward.detach(),
                "rejected_reward": rejected_reward.detach()
            }
        return loss