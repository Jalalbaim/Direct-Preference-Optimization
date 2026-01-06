# src/ppo/trainer.py
import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.dpo.utils import set_seed
from .rewards import load_sentiment_classifier, sentiment_reward
from .utils import (
    generate_with_model,
    response_logprob,
    value_forward,
    save_ppo_checkpoint,
)

class PPOTrainer:
    def __init__(self, model_bundle, value_head, train_loader, config: Dict[str, Any]):
        self.mb = model_bundle
        self.policy = self.mb.policy_model
        self.ref = self.mb.ref_model
        self.tokenizer = self.mb.tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.value_head = value_head

        self.cfg = config
        self.ppo_cfg = config["ppo"]
        self.tr_cfg = config["training"]
        self.log_cfg = config["logging"]
        self.data_cfg = config["data"]
        self.reward_cfg = config["reward"]

        self.train_loader = train_loader

        # Reward model
        self.clf_tok, self.clf = load_sentiment_classifier(self.reward_cfg["clf_model_name"], self.device)

        # Optimizers
        self.policy_optim = AdamW(self.policy.parameters(), lr=float(self.tr_cfg["learning_rate"]))
        self.value_optim = AdamW(self.value_head.parameters(), lr=float(self.tr_cfg["value_learning_rate"]))

        self.grad_accum = int(self.tr_cfg["grad_accumulation_steps"])
        self.max_grad_norm = float(self.tr_cfg["max_grad_norm"])
        self.save_dir = self.log_cfg["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        self.clip_range = float(self.ppo_cfg["clip_range"])
        self.kl_coef = float(self.ppo_cfg["kl_coef"])
        self.vf_coef = float(self.ppo_cfg["vf_coef"])
        self.ppo_epochs = int(self.ppo_cfg["ppo_epochs"])

        self.temperature = float(self.ppo_cfg["temperature"])
        self.top_p = float(self.ppo_cfg["top_p"])

        self.max_new_tokens = int(self.data_cfg["max_new_tokens"])

    def train(self):
        set_seed(int(self.tr_cfg["seed"]))

        num_epochs = int(self.tr_cfg["num_epochs"])
        log_every = int(self.log_cfg["log_every"])
        max_steps = self.tr_cfg.get("max_steps", None)
        max_steps = None if max_steps in (None, "null") else int(max_steps)

        global_step = 0
        running_loss = 0.0

        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        self.policy.train()
        self.value_head.train()

        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f"PPO Epoch {epoch+1}/{num_epochs}")
            for step, prompts in enumerate(pbar):
                # prompts est une liste (batch) ; ici batch_size=1 conseillé
                prompt = prompts[0]

                # 1) rollout: generate response
                response_text, full_ids, prompt_len = generate_with_model(
                    self.policy,
                    self.tokenizer,
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    device=self.device,
                )

                input_ids = full_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)

                # 2) reward + KL (avec logprobs "old" calculés AVANT update)
                with torch.no_grad():
                    old_logp = response_logprob(self.policy, input_ids, attention_mask, prompt_len)  # [1]
                    ref_logp = response_logprob(self.ref, input_ids, attention_mask, prompt_len)     # [1]
                    approx_kl = (old_logp - ref_logp)  # [1]

                    r = sentiment_reward([response_text], self.clf_tok, self.clf, self.device)  # [1]
                    # Convert reward to the same dtype as the model
                    r = r.to(dtype=self.policy.dtype)
                    returns = (r - self.kl_coef * approx_kl).detach()  # [1]

                # 3) PPO update(s) sur cet échantillon
                for _ in range(self.ppo_epochs):
                    new_logp = response_logprob(self.policy, input_ids, attention_mask, prompt_len)  # [1] grad OK
                    ratio = torch.exp(new_logp - old_logp)  # [1]

                    values = value_forward(self.policy, self.value_head, input_ids, attention_mask)  # [1]
                    adv = (returns - values.detach())  # [1]

                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                    policy_loss = -torch.min(unclipped, clipped).mean()

                    value_loss = F.mse_loss(values, returns)

                    loss = policy_loss + self.vf_coef * value_loss
                    loss = loss / self.grad_accum
                    loss.backward()

                    running_loss += float(loss.item())

                # 4) optim step
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)

                    self.policy_optim.step()
                    self.value_optim.step()
                    self.policy_optim.zero_grad(set_to_none=True)
                    self.value_optim.zero_grad(set_to_none=True)

                    global_step += 1
                    if global_step % log_every == 0:
                        pbar.set_postfix({"loss": f"{running_loss / log_every:.4f}"})
                        running_loss = 0.0

                torch.cuda.empty_cache()

                if max_steps is not None and global_step >= max_steps:
                    break

            # 5) checkpoint fin d'epoch
            ckpt_path = os.path.join(self.save_dir, f"policy_epoch_{epoch+1}.pt")
            save_ppo_checkpoint(ckpt_path, self.policy, self.value_head, self.policy_optim, self.value_optim)

            if max_steps is not None and global_step >= max_steps:
                break
