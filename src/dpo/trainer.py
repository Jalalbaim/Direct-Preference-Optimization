import os
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from .losses import dpo_loss
from .models import compute_logprobs, ModelBundle
from .utils import set_seed, save_checkpoint


class DPOTrainer:
    def __init__(
        self,
        model_bundle: ModelBundle,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        self.mb = model_bundle
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = self.mb.device
        self.policy_model = self.mb.policy_model
        self.ref_model = self.mb.ref_model
        self.beta = float(config["dpo"]["beta"])

        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        self.grad_accum = config["training"]["grad_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]

        self.save_dir = config["logging"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        set_seed(self.config["training"]["seed"])

        num_epochs = self.config["training"]["num_epochs"]
        log_every = self.config["logging"]["log_every"]

        global_step = 0

        self.policy_model.train()

        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0

            for step, batch in enumerate(pbar):
                loss = self._train_step(batch)
                running_loss += loss.item()

                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % log_every == 0:
                        avg_loss = running_loss / log_every
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        running_loss = 0.0

            # fin d'epoch = on sauvegarde un checkpoint
            ckpt_path = os.path.join(self.save_dir, f"policy_epoch_{epoch+1}.pt")
            save_checkpoint(self.policy_model, self.optimizer, ckpt_path)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # déplacer batch sur device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # log-probs policy
        policy_chosen_logps = compute_logprobs(
            self.policy_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        policy_rejected_logps = compute_logprobs(
            self.policy_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )

        # log-probs ref
        with torch.no_grad():
            ref_chosen_logps = compute_logprobs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_mask"],
            )
            ref_rejected_logps = compute_logprobs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )

        loss = loss / self.grad_accum
        loss.backward()
        
        # Libérer le cache CUDA pour économiser de la mémoire
        torch.cuda.empty_cache()

        return loss
