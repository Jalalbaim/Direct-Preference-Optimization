import torch
import torch.nn.functional as F


def grpo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    clip_epsilon: float,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO (Group Relative Policy Optimization) loss.

    GRPO génère plusieurs réponses (group_size) par prompt et normalise
    les rewards au sein de chaque groupe. C'est une version simplifiée de PPO
    qui n'utilise pas de value function.

    Args:
        policy_logps: Log-probs de la politique actuelle [B * group_size]
        ref_logps: Log-probs du modèle de référence [B * group_size]
        old_logps: Log-probs de l'ancienne politique [B * group_size]
        rewards: Rewards pour chaque réponse [B * group_size]
        group_size: Nombre de réponses générées par prompt
        clip_epsilon: Epsilon pour le clipping PPO
        beta: Coefficient pour la pénalité KL

    Returns:
        loss: Loss totale
        stats: Dictionnaire avec les statistiques
    """
    # Reshape pour séparer les groupes
    # B = nombre de prompts, G = group_size
    total = len(rewards)
    B = total // group_size
    
    policy_logps = policy_logps.view(B, group_size)
    ref_logps = ref_logps.view(B, group_size)
    old_logps = old_logps.view(B, group_size)
    rewards = rewards.view(B, group_size)

    # Normalisation des rewards au sein de chaque groupe
    # (moyenne 0, std 1 par groupe)
    rewards_mean = rewards.mean(dim=1, keepdim=True)
    rewards_std = rewards.std(dim=1, keepdim=True) + 1e-8
    normalized_rewards = (rewards - rewards_mean) / rewards_std

    # Flatten pour le calcul
    normalized_rewards = normalized_rewards.view(-1)
    policy_logps = policy_logps.view(-1)
    ref_logps = ref_logps.view(-1)
    old_logps = old_logps.view(-1)

    # KL divergence avec le modèle de référence
    kl_div = policy_logps - ref_logps

    # Avantage = reward normalisé - beta * KL
    advantages = normalized_rewards - beta * kl_div

    # Ratio de probabilité (comme PPO)
    ratio = torch.exp(policy_logps - old_logps)

    # Surrogate losses avec clipping
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    # Policy loss
    policy_loss = -torch.min(surr1, surr2).mean()

    # Statistiques
    stats = {
        "loss/total": policy_loss.item(),
        "loss/policy": policy_loss.item(),
        "kl_div": kl_div.mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "rewards_mean": rewards.mean().item(),
        "rewards_std": rewards.std().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }

    return policy_loss, stats


def grpo_loss_simple(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Version simplifiée de GRPO sans clipping PPO.
    Similaire à DPO mais avec normalisation de groupe.

    Args:
        policy_logps: Log-probs de la politique actuelle [B * group_size]
        ref_logps: Log-probs du modèle de référence [B * group_size]
        rewards: Rewards pour chaque réponse [B * group_size]
        group_size: Nombre de réponses générées par prompt
        beta: Coefficient pour la pondération

    Returns:
        loss: Loss totale
        stats: Dictionnaire avec les statistiques
    """
    # Reshape pour séparer les groupes
    total = len(rewards)
    B = total // group_size
    
    policy_logps = policy_logps.view(B, group_size)
    ref_logps = ref_logps.view(B, group_size)
    rewards = rewards.view(B, group_size)

    # Normalisation des rewards au sein de chaque groupe
    rewards_mean = rewards.mean(dim=1, keepdim=True)
    rewards_std = rewards.std(dim=1, keepdim=True) + 1e-8
    normalized_rewards = (rewards - rewards_mean) / rewards_std

    # Log-ratios
    log_ratios = policy_logps - ref_logps

    # Loss: -E[normalized_reward * (log_policy - beta * log_ratio)]
    loss = -(normalized_rewards * (policy_logps - beta * log_ratios)).mean()

    # Statistiques
    stats = {
        "loss/total": loss.item(),
        "kl_div": log_ratios.mean().item(),
        "rewards_mean": rewards.mean().item(),
        "rewards_std": rewards.std().item(),
    }

    return loss, stats
