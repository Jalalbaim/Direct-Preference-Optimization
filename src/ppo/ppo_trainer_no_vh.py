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

from ..core.models import compute_logprobs, ModelBundle
from ..dpo.reward_models import RewardModel
from ..core.utils import set_seed, save_checkpoint


def ppo_loss_no_value(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
    entropy_coef: float,
    entropy: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    PPO loss SANS value head - utilise directement le reward model.
    
    Args:
        policy_logps: Log-probs de la politique actuelle [B]
        ref_logps: Log-probs du modèle de référence [B]
        old_logps: Log-probs de l'ancienne politique [B]
        advantages: Advantages calculés à partir du reward model [B]
        clip_epsilon: Epsilon pour le clipping (e.g., 0.2)
        entropy_coef: Coefficient pour l'entropy bonus
        entropy: Entropy de la distribution [B]
    
    Returns:
        loss: Loss total
        stats: Dictionnaire avec les statistiques
    """
    # KL divergence avec le modèle de référence
    kl_div = (policy_logps - ref_logps).mean()

    # Ratio de probabilité
    ratio = torch.exp(policy_logps - old_logps)

    # Surrogate losses
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # Policy loss (on minimise, donc on prend -min)
    policy_loss = -torch.min(surr1, surr2).mean()

    # Entropy bonus (on veut maximiser l'entropie, donc -entropy pour minimiser)
    entropy_loss = -entropy.mean()

    # Loss totale (PAS de value loss)
    loss = policy_loss + entropy_coef * entropy_loss

    # Statistiques pour logging
    stats = {
        "loss/total": loss.item(),
        "loss/policy": policy_loss.item(),
        "loss/entropy": entropy_loss.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }

    return loss, stats


class PPOTrainerNoValueHead:
    """
    PPO Trainer SANS value head.
    Utilise directement le reward model pour calculer les avantages.
    Plus simple et économise des paramètres.
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

        # PAS de value head - c'est la différence principale !
        print("✓ Mode PPO sans Value Head - rewards directs du reward model")

        # Paramètres PPO
        self.clip_epsilon = float(config["ppo"]["clip_epsilon"])
        self.entropy_coef = float(config["ppo"]["entropy_coef"])
        self.gamma = float(config["ppo"]["gamma"])
        self.num_ppo_epochs = int(config["ppo"]["num_ppo_epochs"])
        
        # Target KL for early stopping
        self.target_kl = float(config["ppo"].get("target_kl", 0.01))
        self.use_kl_early_stop = bool(config["ppo"].get("use_kl_early_stop", True))

        # Reward model
        reward_model_name = config.get("reward_model", {}).get(
            "name", "lvwerra/distilbert-imdb"
        )
        self.reward_model = RewardModel(reward_model_name, device=str(self.device))

        # Optimiseur (SANS value head, moins de paramètres)
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
                project=wandb_cfg.get("wandb_project", "ppo-no-vh"),
                name=wandb_cfg.get("wandb_run_name", config.get("experiment_name", "ppo-no-vh")),
                config=config,
            )

    def train(self):
        set_seed(self.config["training"]["seed"])

        num_epochs = self.config["training"]["num_epochs"]
        log_every = self.config["logging"]["log_every"]

        global_step = 0

        for epoch in range(num_epochs):
            pbar = tqdm(self.prompt_loader,
                        desc=f"PPO (No VH) Epoch {epoch+1}/{num_epochs}")
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

                    if self.use_wandb:
                        wandb.log({**avg_stats, "epoch": epoch, "global_step": global_step}, step=global_step)
                    running_stats = {}

            # Sauvegarder checkpoint
            ckpt_path = os.path.join(self.save_dir,
                                     f"policy_ppo_no_vh_epoch_{epoch+1}.pt")
            save_checkpoint(self.policy_model, self.optimizer, ckpt_path)

            if self.use_wandb:
                wandb.log({"checkpoint_epoch": epoch + 1}, step=global_step)

    def _ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Un step PPO complet SANS value head:
        1. Génération de réponses
        2. Calcul des rewards directs
        3. Advantages = rewards (pas de baseline)
        4. Multiple epochs de PPO sur les données collectées
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 1. Générer des réponses avec la politique actuelle
        generated_data = self._generate_responses(batch)

        # 2. Calculer les rewards DIRECTEMENT
        rewards = self.reward_model.compute_rewards(generated_data["texts"])
        model_dtype = next(self.policy_model.parameters()).dtype
        rewards = rewards.to(dtype=model_dtype)

        # 3. Advantages = rewards directement (pas de value baseline)
        # On peut normaliser pour stabiliser l'entraînement
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

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
            )

            # Entropie réelle sur les tokens de réponse
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_mask = generated_data["response_mask"][:, 1:].contiguous()
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            probs = torch.softmax(shift_logits, dim=-1)
            entropy_tokens = -(probs * log_probs).sum(dim=-1)
            # moyenne sur les positions de réponse
            entropy = (entropy_tokens * shift_mask).sum(dim=-1) / (shift_mask.sum(dim=-1) + 1e-8)

            # Calculer les nouveaux log-probs
            policy_logps = self._compute_logprobs_from_outputs(
                logits,
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

            # Entropy (approximation simple)
            entropy = torch.ones_like(policy_logps) * 0.1  # TODO: calculer vraie entropy

            # Vérifier KL avant de continuer
            approx_kl = (policy_logps - old_logps).mean().item()
            if self.use_kl_early_stop and approx_kl > self.target_kl:
                early_stopped = True
                total_stats["early_stop_epoch"] = [ppo_epoch]
                total_stats["approx_kl_at_stop"] = [approx_kl]
                break

            # PPO loss SANS value head
            loss, stats = ppo_loss_no_value(
                policy_logps,
                ref_logps,
                old_logps,
                advantages,
                self.clip_epsilon,
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
            stats["reward_mean"] = rewards.mean().item()
            stats["reward_std"] = rewards.std().item()
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

        prompt_ids = batch["input_ids"]
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
        full_mask = torch.ones_like(full_ids, dtype=torch.long)

        # Masque de réponse
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
