"""
Script pour visualiser et analyser les différences entre DPO, PPO et GRPO.
Crée des graphiques comparatifs des losses et performances.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_comparison(results: Dict, output_dir: str = "plots"):
    """
    Crée des visualisations comparatives des résultats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraire les données
    methods = list(results.keys())
    mean_rewards = [results[m]['mean_reward'] for m in methods]
    std_rewards = [results[m]['std_reward'] for m in methods]
    
    # Figure 1: Bar plot des rewards moyens
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    
    # Colorer les barres
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Comparison of RLHF Methods: Reward Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/reward_comparison.png")
    plt.close()
    
    # Figure 2: Variance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, std_rewards, alpha=0.7)
    
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Reward Standard Deviation', fontsize=12)
    ax.set_title('Variance in Generated Responses', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    for i, std in enumerate(std_rewards):
        ax.text(i, std + 0.005, f'{std:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variance_comparison.png'), dpi=300)
    print(f"Saved: {output_dir}/variance_comparison.png")
    plt.close()
    
    # Figure 3: Method characteristics (radar chart)
    create_method_characteristics_chart(output_dir)


def create_method_characteristics_chart(output_dir: str):
    """
    Crée un radar chart comparant les caractéristiques des méthodes.
    """
    categories = ['Simplicité', 'Stabilité', 'Qualité', 'Flexibilité', 'Efficacité']
    
    # Scores subjectifs basés sur les caractéristiques connues
    scores = {
        'DPO': [9, 9, 7, 6, 8],      # Simple, stable, qualité ok, moins flexible
        'PPO': [4, 5, 9, 10, 5],     # Complexe, moins stable, haute qualité, très flexible
        'GRPO': [7, 7, 8, 8, 6],     # Compromis entre DPO et PPO
    }
    
    # Nombre de variables
    N = len(categories)
    
    # Angles pour chaque axe
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialiser le plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Tracer chaque méthode
    colors = {'DPO': '#FF6B6B', 'PPO': '#4ECDC4', 'GRPO': '#45B7D1'}
    for method, score_list in scores.items():
        values = score_list + score_list[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[method])
        ax.fill(angles, values, alpha=0.15, color=colors[method])
    
    # Fixer les labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], size=10)
    ax.grid(True)
    
    # Titre et légende
    ax.set_title('Caractéristiques des Méthodes RLHF', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_characteristics.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/method_characteristics.png")
    plt.close()


def create_architecture_comparison_table():
    """
    Crée un tableau HTML comparant les architectures.
    """
    html = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .yes { color: green; font-weight: bold; }
            .no { color: red; font-weight: bold; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h2>Comparison of RLHF Methods</h2>
        <table>
            <tr>
                <th>Caractéristique</th>
                <th>DPO</th>
                <th>PPO</th>
                <th>GRPO</th>
            </tr>
            <tr>
                <td>Type d'apprentissage</td>
                <td>Offline</td>
                <td>Online</td>
                <td>Online</td>
            </tr>
            <tr>
                <td>Données requises</td>
                <td>Paires de préférences</td>
                <td>Prompts</td>
                <td>Prompts</td>
            </tr>
            <tr>
                <td>Reward model</td>
                <td class="no">Non</td>
                <td class="yes">Oui</td>
                <td class="yes">Oui</td>
            </tr>
            <tr>
                <td>Value function</td>
                <td class="no">Non</td>
                <td class="yes">Oui</td>
                <td class="no">Non</td>
            </tr>
            <tr>
                <td>Génération requise</td>
                <td class="no">Non</td>
                <td class="yes">Oui</td>
                <td class="yes">Oui (multiple)</td>
            </tr>
            <tr>
                <td>Complexité</td>
                <td>Faible</td>
                <td>Élevée</td>
                <td>Moyenne</td>
            </tr>
            <tr>
                <td>Stabilité</td>
                <td>Élevée</td>
                <td>Moyenne</td>
                <td>Élevée</td>
            </tr>
            <tr>
                <td>Mémoire GPU</td>
                <td>Faible</td>
                <td>Élevée</td>
                <td>Moyenne</td>
            </tr>
            <tr>
                <td>Temps d'entraînement</td>
                <td>Rapide</td>
                <td>Lent</td>
                <td>Moyen</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    with open("plots/architecture_comparison.html", "w") as f:
        f.write(html)
    print("Saved: plots/architecture_comparison.html")


def main():
    # Charger les résultats
    results_file = "comparison_results.json"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        print("Please run 'python scripts/compare_methods.py' first.")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    print("Creating visualization plots...")
    plot_comparison(results)
    
    print("\nCreating architecture comparison table...")
    create_architecture_comparison_table()
    
    print("\n✅ All visualizations created successfully!")
    print("Check the 'plots/' directory for output files.")


if __name__ == "__main__":
    # Check if matplotlib is installed
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Error: matplotlib not installed.")
        print("Install with: pip install matplotlib")
        exit(1)
    
    main()
