"""
Script pour extraire les prompts des données de préférences
et créer un fichier prompts.jsonl pour PPO/GRPO.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


def extract_prompts(preference_file: str, output_file: str):
    """
    Extrait les prompts uniques d'un fichier de préférences.
    """
    prompts = set()
    
    with open(preference_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.add(obj["prompt"])
    
    # Écrire les prompts
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in sorted(prompts):
            json.dump({"prompt": prompt}, f)
            f.write("\n")
    
    print(f"Extracted {len(prompts)} unique prompts")
    print(f"Saved to: {output_file}")


def main():
    # Chemins
    train_prefs = "data/processed/sentiment/train_preferences.jsonl"
    val_prefs = "data/processed/sentiment/val_preferences.jsonl"
    
    output_file = "data/processed/sentiment/prompts.jsonl"
    
    print("Extracting prompts from preference files...")
    
    # Combiner train et val pour avoir tous les prompts
    all_prompts = set()
    
    for pref_file in [train_prefs, val_prefs]:
        if os.path.exists(pref_file):
            with open(pref_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    all_prompts.add(obj["prompt"])
    
    # Écrire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in sorted(all_prompts):
            json.dump({"prompt": prompt}, f)
            f.write("\n")
    
    print(f"Extracted {len(all_prompts)} unique prompts")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
