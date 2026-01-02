import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List


class RewardModel:
    """
    Modèle de reward basé sur un classifieur de sentiment.
    Pour PPO/GRPO, on a besoin de calculer des rewards pour des réponses générées.
    """

    def __init__(self, model_name: str = "lvwerra/distilbert-imdb", device: str = None):
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Charger un modèle de sentiment pré-entraîné
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        """
        Calcule les rewards pour une liste de textes.
        
        Args:
            texts: Liste de textes générés
            
        Returns:
            rewards: Tensor [len(texts)] avec un reward par texte
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits  # [B, num_labels]

        # Pour un modèle de sentiment binaire (négatif/positif)
        # On utilise le score du label positif comme reward
        if logits.shape[1] == 2:
            rewards = torch.softmax(logits, dim=-1)[:, 1]  # probabilité du label positif
        else:
            # Pour un modèle multi-classe, utiliser le score max
            rewards = torch.softmax(logits, dim=-1).max(dim=-1).values

        return rewards


class ValueHead(nn.Module):
    """
    Value head pour PPO.
    Prend les hidden states d'un LM et prédit une valeur scalaire.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, dtype="bfloat16")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, H] ou [B, H]
        Returns:
            values: [B] ou [B, L]
        """
        values = self.linear(hidden_states).squeeze(-1)
        return values


def add_value_head_to_model(model: nn.Module) -> nn.Module:
    """
    Ajoute un value head à un modèle de langage.
    
    Args:
        model: Modèle de langage (AutoModelForCausalLM)
        
    Returns:
        model avec value_head ajouté
    """
    # Obtenir la taille des hidden states
    hidden_size = model.config.hidden_size
    
    # Créer et ajouter le value head
    value_head = ValueHead(hidden_size)
    
 
    model.value_head = value_head
    
    return model
