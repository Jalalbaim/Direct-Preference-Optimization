# Direct Preference Optimization (DPO)

Implementation of the DPO algorithm for language model alignment based on human preferences.

## Models

- **y_w, y_l generation**: Ollama gemma3:4b (preference pairs creation)
- **SFT Model**: TinyLlama-1.1B-Chat-v1.0
- **Method**: DPO Direct preference optimization

## Features

The project implements DPO training on a sentiment classification task with:

- Preference data preparation (chosen/rejected pairs)
- Training with DPO loss (β=0.1)
- Aligned model performance evaluation

## Usage

```bash
#Install requirements
pip install -r requirements.txt
```

### Sentiment generation

```bash
# Prepare data
python scripts/prepare_sentiment_data.py

# Train model
python scripts/train_sentiment.py
python3 scripts/train_sentiment.py

# Evaluate
python scripts/eval_sentiment.py
```

### Summarization
```bash
# Prepare data
python3 scripts/summarization/prepare_summary_data.py


```
