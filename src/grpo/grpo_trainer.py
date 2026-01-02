import os
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from .grpo_losses import grpo_loss
from ..dpo.models import ModelBundle
from ..dpo.reward_models import RewardModel
from ..dpo.utils import set_seed, save_checkpoint


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer.
    
    GRPO génère plusieurs réponses par prompt (group_size),
    calcule les rewards, et normalise au sein de chaque groupe.
    Plus simple que PPO car pas de value function.
    """

    def __init__(
        self,
        model_bundle: ModelBundle,
        prompt_loader: DataLoader,
        config: Dict[str, Any],
    ):
        self.mb = model_bundle
        self.prompt_loader = prompt_loader
        self.config = config

        self.device = self.mb.device
        self.policy_model = self.mb.policy_model
        self.ref_model = self.mb.ref_model

        # Paramètres GRPO
        self.group_size = int(config["grpo"]["group_size"])
        self.clip_epsilon = float(config["grpo"]["clip_epsilon"])
        self.beta = float(config["grpo"]["beta"])
        self.num_grpo_epochs = int(config["grpo"]["num_grpo_epochs"])

        # Reward model
        reward_model_name = config.get("reward_model", {}).get(
            "name", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.reward_model = RewardModel(reward_model_name)

        # Optimiseur
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.max_gen_length = config["generation"]["max_length"]
        self.temperature = config["generation"]["temperature"]

        self.save_dir = config["logging"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        set_seed(self.config["training"]["seed"])

        num_epochs = self.config["training"]["num_epochs"]
        log_every = self.config["logging"]["log_every"]

        global_step = 0

        for epoch in range(num_epochs):
            pbar = tqdm(self.prompt_loader, desc=f"GRPO Epoch {epoch+1}/{num_epochs}")
            running_stats = {}

            for step, batch in enumerate(pbar):
                stats = self._grpo_step(batch)
                
                # Accumulate stats
                for k, v in stats.items():
                    if k not in running_stats:
                        running_stats[k] = []
                    running_stats[k].append(v)

                global_step += 1

                if global_step % log_every == 0:
                    avg_stats = {k: sum(v) / len(v) for k, v in running_stats.items()}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in avg_stats.items()})
                    running_stats = {}

            # Sauvegarder checkpoint
            ckpt_path = os.path.join(self.save_dir, f"policy_grpo_epoch_{epoch+1}.pt")
            save_checkpoint(self.policy_model, self.optimizer, ckpt_path)

    def _grpo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Un step GRPO complet:
        1. Génération de group_size réponses par prompt
        2. Calcul des rewards
        3. Multiple epochs de GRPO
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 1. Générer group_size réponses par prompt
        generated_data = self._generate_group_responses(batch)

        # 2. Calculer les rewards
        rewards = self.reward_model.compute_rewards(generated_data["texts"])

        # 3. Stocker les log-probs de l'ancienne politique
        old_logps = generated_data["logps"].detach()

        # 4. Multiple epochs de GRPO
        total_stats = {}
        for grpo_epoch in range(self.num_grpo_epochs):
            # Forward avec la nouvelle politique
            policy_logps = self._compute_logprobs(
                self.policy_model,
                generated_data["input_ids"],
                generated_data["attention_mask"],
                generated_data["response_mask"],
            )

            # Log-probs du modèle de référence
            with torch.no_grad():
                ref_logps = self._compute_logprobs(
                    self.ref_model,
                    generated_data["input_ids"],
                    generated_data["attention_mask"],
                    generated_data["response_mask"],
                )

            # GRPO loss
            loss, stats = grpo_loss(
                policy_logps,
                ref_logps,
                old_logps,
                rewards,
                self.group_size,
                self.clip_epsilon,
                self.beta,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate stats
            for k, v in stats.items():
                if k not in total_stats:
                    total_stats[k] = []
                total_stats[k].append(v)

        # Average stats over GRPO epochs
        avg_stats = {k: sum(v) / len(v) for k, v in total_stats.items()}
        return avg_stats

    @torch.no_grad()
    def _generate_group_responses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Génère group_size réponses pour chaque prompt.
        """
        self.policy_model.eval()

        prompt_ids = batch["input_ids"]  # [B, L_prompt]
        prompt_mask = batch["attention_mask"]
        B = prompt_ids.shape[0]

        # Répéter chaque prompt group_size fois
        # [B, L] -> [B * group_size, L]
        prompt_ids_expanded = prompt_ids.repeat_interleave(self.group_size, dim=0)
        prompt_mask_expanded = prompt_mask.repeat_interleave(self.group_size, dim=0)

        # Génération
        generated_ids = self.policy_model.generate(
            input_ids=prompt_ids_expanded,
            attention_mask=prompt_mask_expanded,
            max_new_tokens=self.max_gen_length,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.mb.tokenizer.pad_token_id,
        )

        # Extraire les réponses
        L_prompt = prompt_ids.shape[1]
        response_ids = generated_ids[:, L_prompt:]
        
        # Tenseurs complets
        full_ids = generated_ids
        full_mask = torch.ones_like(full_ids)

        # Masque de réponse
        response_mask = torch.zeros_like(full_ids)
        response_mask[:, L_prompt:] = 1

        # Décoder les textes
        texts = self.mb.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Calculer les log-probs avec la politique actuelle
        logps = self._compute_logprobs(
            self.policy_model,
            full_ids,
            full_mask,
            response_mask,
        )

        self.policy_model.train()

        return {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "response_mask": response_mask,
            "texts": texts,
            "logps": logps,
        }

    def _compute_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule les log-probs des réponses.
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Décaler
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()

        # Log-probs
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Masquer et sommer
        token_logps = token_logps * shift_mask
        seq_logps = token_logps.sum(dim=-1)

        return seq_logps
