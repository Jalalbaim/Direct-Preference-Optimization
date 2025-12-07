import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    DPO loss, as in the paper.

    Inputs are log-probs summed (ou moyennés) sur les tokens de la réponse,
    shape: (batch,).

    L = - E[ log σ( β * ( (logπ_c - logπ_r) - (logπ_ref_c - logπ_ref_r) ) ) ]
    """

    # [B] log-ratio policy
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # [B] log-ratio ref
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # [B] logits dpo
    logits = beta * (pi_logratios - ref_logratios)

    loss = -F.logsigmoid(logits).mean()
    return loss
