import os
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not available. Using standard AdamW optimizer.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging to console only.")

from .ppo_losses import ppo_loss, compute_gae
from ..core.models import compute_logprobs, ModelBundle
from ..dpo.reward_models import RewardModel, add_value_head_to_model
from ..core.utils import set_seed, save_checkpoint


class PPOTrainer:
    """
    PPO Trainer pour l'entraînement avec génération et reward model.
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

        # Activer gradient checkpointing pour économiser la mémoire
        use_gradient_checkpointing = config.get("training", {}).get(
            "gradient_checkpointing", True
        )
        if use_gradient_checkpointing and hasattr(self.policy_model, 'gradient_checkpointing_enable'):
            self.policy_model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing activé (économie ~30-40% d'activations)")
        
        # Ajouter un value head au policy model pour PPO
        self.policy_model = add_value_head_to_model(self.policy_model)
        # S'assurer que le modèle est sur le bon device
        self.policy_model = self.policy_model.to(self.device)

        # Paramètres PPO
        self.clip_epsilon = float(config["ppo"]["clip_epsilon"])
        self.value_coef = float(config["ppo"]["value_coef"])
        self.entropy_coef = float(config["ppo"]["entropy_coef"])
        self.gamma = float(config["ppo"]["gamma"])
        self.gae_lambda = float(config["ppo"]["gae_lambda"])
        self.num_ppo_epochs = int(config["ppo"]["num_ppo_epochs"])
        
        # Target KL for early stopping
        self.target_kl = float(config["ppo"].get("target_kl", 0.01))
        self.use_kl_early_stop = bool(config["ppo"].get("use_kl_early_stop", True))

        # Reward model
        reward_model_name = config.get("reward_model", {}).get(
            "name", "lvwerra/distilbert-imdb"
        )
        self.reward_model = RewardModel(reward_model_name, device=str(self.device))

        # Optimiseur (inclut le value head)
        use_8bit_optimizer = config.get("training", {}).get(
            "use_8bit_optimizer", True
        )
        
        if use_8bit_optimizer and BNB_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy_model.parameters(),
                lr=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
            )
            print("✓ Optimiseur 8-bit activé (économie ~50% sur états optimiseur)")
        else:
            self.optimizer = AdamW(
                self.policy_model.parameters(),
                lr=float(config["training"]["learning_rate"]),
                weight_decay=float(config["training"]["weight_decay"]),
            )
            if use_8bit_optimizer and not BNB_AVAILABLE:
                print("⚠ Optimiseur 8-bit demandé mais bitsandbytes non disponible")

        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.max_gen_length = config["generation"]["max_length"]
        self.temperature = config["generation"]["temperature"]

        self.save_dir = config["logging"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # Weights & Biases (optionnel)
        wandb_cfg = config.get("logging", {})
        self.use_wandb = bool(wandb_cfg.get("wandb_enabled", False)) and WANDB_AVAILABLE
        if wandb_cfg.get("wandb_enabled", False) and not WANDB_AVAILABLE:
            print("⚠ wandb_enabled=True mais wandb n'est pas installé — aucun log wandb")

        if self.use_wandb:
            wandb.init(
                project=wandb_cfg.get("wandb_project", "ppo-training"),
                name=wandb_cfg.get("wandb_run_name", config.get("experiment_name", "ppo-run")),
                config=config,
            )

    def train(self):
        set_seed(self.config["training"]["seed"])

        num_epochs = self.config["training"]["num_epochs"]
        log_every = self.config["logging"]["log_every"]

        global_step = 0

        for epoch in range(num_epochs):
            pbar = tqdm(self.prompt_loader,
                        desc=f"PPO Epoch {epoch+1}/{num_epochs}")
            running_stats = {}

            for step, batch in enumerate(pbar):
                stats = self._ppo_step(batch)
                
                # Accumulate stats
                for k, v in stats.items():
                    if k not in running_stats:
                        running_stats[k] = []
                    running_stats[k].append(v)

                global_step += 1

                if global_step % log_every == 0:
                    avg_stats = {k: sum(v) / len(v) for k, v in
                                 running_stats.items()}
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in
                                      avg_stats.items()})

                    # Log vers wandb si activé
                    if self.use_wandb:
                        wandb.log({**avg_stats, "epoch": epoch, "global_step": global_step}, step=global_step)
                    running_stats = {}

            # Sauvegarder checkpoint
            ckpt_path = os.path.join(self.save_dir,
                                     f"policy_ppo_epoch_{epoch+1}.pt")
            save_checkpoint(self.policy_model, self.optimizer, ckpt_path)

            if self.use_wandb:
                wandb.log({"checkpoint_epoch": epoch + 1}, step=global_step)

    def _ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Un step PPO complet:
        1. Génération de réponses
        2. Calcul des rewards
        3. Multiple epochs de PPO sur les données collectées
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 1. Générer des réponses avec la politique actuelle
        generated_data = self._generate_responses(batch)

        # 2. Calculer les rewards
        rewards = self.reward_model.compute_rewards(generated_data["texts"])
        # Convertir rewards au dtype du policy model
        model_dtype = next(self.policy_model.parameters()).dtype
        rewards = rewards.to(dtype=model_dtype)

        # 3. Calculer les values et advantages
        with torch.no_grad():
            # Re-forward pour obtenir les values de l'ancienne politique
            old_outputs = self.policy_model(
                input_ids=generated_data["input_ids"],
                attention_mask=generated_data["attention_mask"],
                output_hidden_states=True,
            )
            # Value à partir du dernier token de prompt
            last_hidden = old_outputs.hidden_states[-1][:, -1, :]  # [B, H]
            old_values = self.policy_model.value_head(last_hidden)  # [B]

            # Compute advantages (simplifié: on utilise juste le reward)
            # Dans une vraie implémentation, on ferait GAE sur une trajectoire
            advantages = rewards - old_values
            returns = rewards

        # 4. Stocker les log-probs de l'ancienne politique
        old_logps = generated_data["logps"].detach()

        # 5. Multiple epochs de PPO avec early stopping basé sur KL
        total_stats = {}
        early_stopped = False
        
        for ppo_epoch in range(self.num_ppo_epochs):
            # Forward avec la nouvelle politique
            outputs = self.policy_model(
                input_ids=generated_data["input_ids"],
                attention_mask=generated_data["attention_mask"],
                output_hidden_states=True,
            )

            # Calculer les nouveaux log-probs
            policy_logps = self._compute_logprobs_from_outputs(
                outputs.logits,
                generated_data["input_ids"],
                generated_data["response_mask"],
            )

            # Calculer les log-probs du modèle de référence
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=generated_data["input_ids"],
                    attention_mask=generated_data["attention_mask"],
                )
                ref_logps = self._compute_logprobs_from_outputs(
                    ref_outputs.logits,
                    generated_data["input_ids"],
                    generated_data["response_mask"],
                )

            # Values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            values = self.policy_model.value_head(last_hidden)

            # Entropy (approximation simple)
            entropy = torch.ones_like(policy_logps) * 0.1  # TODO: calculer vraie entropy

            # Vérifier KL avant de continuer
            approx_kl = (policy_logps - old_logps).mean().item()
            if self.use_kl_early_stop and approx_kl > self.target_kl:
                early_stopped = True
                total_stats["early_stop_epoch"] = [ppo_epoch]
                total_stats["approx_kl_at_stop"] = [approx_kl]
                break

            # PPO loss
            loss, stats = ppo_loss(
                policy_logps,
                ref_logps,
                old_logps,
                advantages,
                values,
                returns,
                self.clip_epsilon,
                self.value_coef,
                self.entropy_coef,
                entropy,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),
                                        self.max_grad_norm)
            self.optimizer.step()

            # Accumulate stats
            stats["approx_kl"] = approx_kl
            for k, v in stats.items():
                if k not in total_stats:
                    total_stats[k] = []
                total_stats[k].append(v)

        # Average stats over PPO epochs
        avg_stats = {k: sum(v) / len(v) for k, v in total_stats.items()}
        return avg_stats

    @torch.no_grad()
    def _generate_responses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Génère des réponses à partir des prompts.
        """
        self.policy_model.eval()

        prompt_ids = batch["input_ids"]  # [B, L_prompt]
        prompt_mask = batch["attention_mask"]

        # Génération
        generated_ids = self.policy_model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=self.max_gen_length,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.mb.tokenizer.pad_token_id,
        )

        # Extraire seulement les tokens générés
        response_ids = generated_ids[:, prompt_ids.shape[1]:]
        
        # Créer les tenseurs complets (prompt + response)
        full_ids = generated_ids
        # Créer attention mask avec le bon dtype (float pour le modèle)
        full_mask = torch.ones_like(full_ids, dtype=torch.long)

        # Masque de réponse (utilisé pour les calculs, doit être en float)
        model_dtype = next(self.policy_model.parameters()).dtype
        response_mask = torch.zeros_like(full_ids, dtype=model_dtype)
        response_mask[:, prompt_ids.shape[1]:] = 1

        # Décoder les textes
        texts = self.mb.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Calculer les log-probs avec la politique actuelle
        policy_outputs = self.policy_model(
            input_ids=full_ids,
            attention_mask=full_mask,
        )
        logps = self._compute_logprobs_from_outputs(
            policy_outputs.logits,
            full_ids,
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

    def _compute_logprobs_from_outputs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule les log-probs à partir des logits.
        """
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
