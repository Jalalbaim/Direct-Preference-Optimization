# RLHF Methods Comparison: DPO vs PPO vs GRPO

Implémentation et comparaison de trois méthodes d'alignement pour les modèles de langage :

- **DPO** (Direct Preference Optimization) - Offline learning from preferences
- **PPO** (Proximal Policy Optimization) - Online RL with reward model
- **GRPO** (Group Relative Policy Optimization) - Simplified PPO with group normalization

## Models

- **y_w, y_l generation**: Ollama gemma3:4b (preference pairs creation)
- **SFT Model**: TinyLlama-1.1B-Chat-v1.0
- **Methods**: DPO, PPO, GRPO
- **Reward Model**: DistilBERT sentiment classifier

## Features

This project implements three state-of-the-art alignment methods:

### DPO (Direct Preference Optimization)
- ✅ Preference data preparation (chosen/rejected pairs)
- ✅ Training with DPO loss (β=0.1)
- ✅ No reward model needed
- ✅ Simple and stable

### PPO (Proximal Policy Optimization)
- ✅ Online generation with reward feedback
- ✅ Value function for advantage estimation
- ✅ Clipped surrogate objective
- ✅ Entropy bonus for exploration

### GRPO (Group Relative Policy Optimization)
- ✅ Group-based response generation
- ✅ Reward normalization within groups
- ✅ No value function needed
- ✅ Reduced variance

## Quick Start

```bash
# Install requirements
pip install -r requirements.txt

# Prepare data
python scripts/prepare_sentiment_data.py  # For DPO
python scripts/prepare_prompts.py         # For PPO/GRPO

# Train with each method
python scripts/train_sentiment.py  # DPO
python scripts/train_ppo.py        # PPO
python scripts/train_grpo.py       # GRPO

# Compare results
python scripts/compare_methods.py
```

## Documentation

- 📘 [**QUICKSTART.md**](QUICKSTART.md) - Guide de démarrage rapide
- 📗 [**COMPARISON_GUIDE.md**](COMPARISON_GUIDE.md) - Guide détaillé de comparaison

## Project Structure

```
├── src/dpo/
│   ├── losses.py           # DPO loss
│   ├── ppo_losses.py       # PPO losses
│   ├── grpo_losses.py      # GRPO losses
│   ├── reward_models.py    # Reward models
│   ├── trainer.py          # DPO trainer
│   ├── ppo_trainer.py      # PPO trainer
│   └── grpo_trainer.py     # GRPO trainer
├── configs/
│   ├── sentiment.yaml      # DPO config
│   ├── ppo_sentiment.yaml  # PPO config
│   └── grpo_sentiment.yaml # GRPO config
└── scripts/
    ├── train_sentiment.py  # Train DPO
    ├── train_ppo.py        # Train PPO
    ├── train_grpo.py       # Train GRPO
    └── compare_methods.py  # Compare all methods
```

## Evaluation

Run the comparison script to evaluate all three methods:

```bash
python scripts/compare_methods.py
```

This will generate a detailed comparison with:
- Mean reward scores
- Standard deviation
- Sample generations
- Results saved to `comparison_results.json`
