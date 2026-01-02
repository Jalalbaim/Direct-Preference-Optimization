import torch
import torch.nn.functional as F


def ppo_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    entropy: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    PPO loss avec clipping, value loss et entropy bonus.

    Args:
        policy_logps: Log-probs de la politique actuelle [B]
        ref_logps: Log-probs du modèle de référence [B]
        old_logps: Log-probs de l'ancienne politique [B]
        advantages: Advantages estimés [B]
        values: Valeurs prédites par le value head [B]
        returns: Returns calculés (rewards-to-go) [B]
        clip_epsilon: Epsilon pour le clipping (e.g., 0.2)
        value_coef: Coefficient pour la value loss
        entropy_coef: Coefficient pour l'entropy bonus
        entropy: Entropy de la distribution [B]

    Returns:
        loss: Loss total
        stats: Dictionnaire avec les statistiques
    """
    # KL divergence avec le modèle de référence (optionnel pour régularisation)
    kl_div = (policy_logps - ref_logps).mean()

    # Ratio de probabilité
    ratio = torch.exp(policy_logps - old_logps)

    # Surrogate losses
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon,
                        1.0 + clip_epsilon) * advantages

    # Policy loss (on minimise, donc on prend -min)
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (MSE entre valeurs prédites et returns)
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (on veut maximiser l'entropie, donc -entropy pour minimiser)
    entropy_loss = -entropy.mean()

    # Loss totale
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Statistiques pour logging
    stats = {
        "loss/total": loss.item(),
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/entropy": entropy_loss.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }

    return loss, stats


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards [B, T]
        values: Values prédites [B, T]
        dones: Indicateurs de fin d'épisode [B, T]
        gamma: Discount factor
        gae_lambda: Lambda pour GAE

    Returns:
        advantages: Advantages [B, T]
        returns: Returns (rewards-to-go) [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    last_gae = 0
    last_value = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        # TD error
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]
                                                      ) - values[:, t]

        # GAE
        advantages[:, t] = last_gae = (
            delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
        )

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns
