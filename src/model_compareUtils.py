import os
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#---
def create_model_comparison_table(m1_result, m2_result, m3_result, m4_result):
    """
    Create a comparison table for the models.
    """
    results_df = pd.DataFrame([m1_result, m2_result, m3_result, m4_result])
    results_df = results_df.set_index('Model')
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    #results_df.reset_index(drop=True, inplace=True)

    # save the comparison table to a JSON file
    os.makedirs('artifacts/compare', exist_ok=True)
    save_path = 'artifacts/compare/model_comparison.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\nModel Comparison Table:\n")
    print(results_df.to_string())

    # save to JSON
    results_df.to_json(save_path, orient='index', indent=4)

    print(f"\n✅ Saved: {save_path}")


    # ── Bar chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    bars = axes[0].barh(results_df.index, results_df['Accuracy'],
                        color=bar_colors, edgecolor='black', height=0.5)
    axes[0].set_title('Validation Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_xlim(85, 100)
    axes[0].grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, results_df['Accuracy']):
        axes[0].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontweight='bold', fontsize=10)

    bars2 = axes[1].barh(results_df.index, results_df['Training Time'],
                        color=bar_colors, edgecolor='black', height=0.5)
    axes[1].set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, results_df['Training Time']):
        axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}s', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('artifacts/compare/model_comparison.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: artifacts/compare/model_comparison.png')
    return results_df



def plot_confusion_matrix(y_val, best_preds, best_classical):
    """
    Plot confusion matrix for the best classical model.
    """
    

    cm = confusion_matrix(y_val, best_preds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.5, linecolor='gray')
    axes[0].set_title(f'Confusion Matrix — {best_classical}\n(Raw Counts)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Normalized (row-wise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Confusion Matrix — {best_classical}\n(Normalized)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('artifacts/compare/confusion_matrix.png', dpi=150, bbox_inches='tight')
    #plt.show()
    print('✅ Saved: artifacts/compare/confusion_matrix.png')

    # ── Per-class accuracy ─────────────────────────────────────────────
    print('\nPer-Digit Accuracy:')
    for i in range(10):
        per_acc = cm[i, i] / cm[i].sum() * 100
        print(f'  Digit {i}: {per_acc:.2f}%')
        