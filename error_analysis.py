import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = '#FFFFFF'

from src.data_loader import load_data
from src.preprocessing import preprocess_data


def load_all_models(model_dir='models'):
    """Load all saved models"""
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        model_path = os.path.join(model_dir, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"Loaded: {model_name}")
    
    return models


def find_common_errors(models, X_test, y_test):
    """Find samples that ALL models misclassify"""
    all_predictions = {}
    all_probabilities = {}
    
    for name, model_data in models.items():
        model = model_data['model']
        scaler = model_data['scaler']
        
        if scaler is not None:
            X_pred = scaler.transform(X_test)
        else:
            X_pred = X_test
        
        y_pred = model.predict(X_pred)
        y_prob = model.predict_proba(X_pred)
        
        all_predictions[name] = y_pred
        all_probabilities[name] = y_prob
    
    error_mask = np.ones(len(y_test), dtype=bool)
    for name, y_pred in all_predictions.items():
        error_mask &= (y_pred != y_test.values)
    
    error_indices = np.where(error_mask)[0]
    print(f"\nTotal samples: {len(y_test)}")
    print(f"Common error samples (all models wrong): {len(error_indices)}")
    print(f"Common error rate: {len(error_indices) / len(y_test) * 100:.2f}%")
    
    return error_indices, all_predictions, all_probabilities


def select_samples(error_indices, all_probabilities, n_samples=5):
    """Select representative samples with different characteristics"""
    if len(error_indices) == 0:
        print("No common error samples found!")
        return []
    
    selected_indices = []
    
    if len(error_indices) <= n_samples:
        selected_indices = error_indices.tolist()
    else:
        avg_confidence = {}
        for idx in error_indices:
            confidences = []
            for name, y_prob in all_probabilities.items():
                prob_correct = y_prob[idx, 0]
                confidences.append(prob_correct)
            avg_confidence[idx] = np.mean(confidences)
        
        sorted_indices = sorted(avg_confidence.items(), key=lambda x: x[1])
        
        n_high = n_samples // 2
        n_low = n_samples - n_high
        
        selected_indices.extend([idx for idx, _ in sorted_indices[:n_high]])
        selected_indices.extend([idx for idx, _ in sorted_indices[-n_low:]])
    
    return selected_indices[:n_samples]


def visualize_sample(idx, X_test, y_test, all_predictions, all_probabilities, feature_names, sample_num):
    """Visualize a single misclassified sample"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    
    actual_label = y_test.iloc[idx]
    
    ax1 = fig.add_subplot(gs[0, :2])
    
    sample_features = X_test.iloc[idx].values
    x_pos = np.arange(len(feature_names))
    
    colors = ['#2196F3' if val > 0 else '#f44336' for val in sample_features]
    bars = ax1.bar(x_pos, sample_features, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Feature Value', fontsize=12)
    ax1.set_title(f'Sample #{idx} - Feature Values', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, sample_features):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    info_text = f"Sample Index: {idx}\n\n"
    info_text += f"Actual Label: {'Disease (1)' if actual_label == 1 else 'No Disease (0)'}\n\n"
    info_text += "Feature Details:\n"
    for fname, fval in zip(feature_names, sample_features):
        info_text += f"  {fname}: {fval:.2f}\n"
    
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3 = fig.add_subplot(gs[1, :])
    
    model_names = list(all_predictions.keys())
    x_models = np.arange(len(model_names))
    width = 0.35
    
    prob_class_0 = []
    prob_class_1 = []
    predictions = []
    
    for name in model_names:
        y_prob = all_probabilities[name]
        prob_class_0.append(y_prob[idx, 0])
        prob_class_1.append(y_prob[idx, 1])
        predictions.append(all_predictions[name][idx])
    
    bars0 = ax3.bar(x_models - width/2, prob_class_0, width, label='P(Class 0)', 
                    color='#4CAF50', alpha=0.8, edgecolor='black')
    bars1 = ax3.bar(x_models + width/2, prob_class_1, width, label='P(Class 1)', 
                    color='#FF5722', alpha=0.8, edgecolor='black')
    
    for i, (bar0, bar1, pred) in enumerate(zip(bars0, bars1, predictions)):
        correct = '✓' if pred == actual_label else '✗'
        color = 'green' if pred == actual_label else 'red'
        
        ax3.text(bar0.get_x() + bar0.get_width()/2., bar0.get_height() + 0.01,
                f'{prob_class_0[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                f'{prob_class_1[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.text(i, -0.08, f'Pred: {pred} {correct}', ha='center', va='top', 
                fontsize=9, color=color, fontweight='bold')
    
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Model Predictions for Sample #{idx} (All Models Misclassified)', 
                 fontsize=14, fontweight='bold', color='red')
    ax3.set_xticks(x_models)
    ax3.set_xticklabels([name.replace('_', ' ') for name in model_names], rotation=15, ha='right')
    ax3.legend(fontsize=11)
    ax3.set_ylim(0, 1.15)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle(f'Error Analysis - Sample {sample_num}', fontsize=16, fontweight='bold', y=0.98)
    
    os.makedirs('plots/error_analysis', exist_ok=True)
    plt.savefig(f'plots/error_analysis/sample_{idx}.png', bbox_inches='tight')
    plt.close()
    
    print(f"Sample #{idx} Visualization Saved")
    print(f"Actual Label: {actual_label}")
    print(f"\nModel Predictions:")
    for name in model_names:
        pred = all_predictions[name][idx]
        prob_0 = all_probabilities[name][idx, 0]
        prob_1 = all_probabilities[name][idx, 1]
        status = 'CORRECT' if pred == actual_label else 'WRONG'
        print(f"  {name:25s}: Pred={pred}, P(0)={prob_0:.4f}, P(1)={prob_1:.4f} [{status}]")


def create_summary_plot(selected_indices, X_test, y_test, all_predictions, all_probabilities, feature_names):
    """Create a summary plot showing all selected samples"""
    n_samples = len(selected_indices)
    if n_samples == 0:
        return
    
    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]
    
    model_names = list(all_predictions.keys())
    
    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        actual_label = y_test.iloc[idx]
        
        x_models = np.arange(len(model_names))
        width = 0.35
        
        prob_class_1 = []
        predictions = []
        
        for name in model_names:
            y_prob = all_probabilities[name]
            prob_class_1.append(y_prob[idx, 1])
            predictions.append(all_predictions[name][idx])
        
        colors = ['red' if pred != actual_label else 'green' for pred in predictions]
        
        bars = ax.bar(x_models, prob_class_1, width, color=colors, alpha=0.7, edgecolor='black')
        
        for j, (bar, prob, pred) in enumerate(zip(bars, prob_class_1, predictions)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{prob:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(j, -0.08, f'{pred}', ha='center', va='top', fontsize=9, 
                   color=colors[j], fontweight='bold')
        
        ax.set_title(f'Sample #{idx}\nActual: {actual_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_models)
        ax.set_xticklabels([name.replace('_', '\n') for name in model_names], fontsize=8)
        ax.set_ylabel('P(Class 1)', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Summary: All Models Misclassified Samples (Red=Wrong, Green=Correct)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/error_analysis/summary_all_samples.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    print("\n[1] Loading data...")
    df = load_data('cardio_train.csv')
    df = preprocess_data(df)
    print(f"Data shape after preprocessing: {df.shape}")
    
    print("\n[2] Splitting data...")
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test set size: {len(y_test)}")
    
    print("\n[3] Loading models...")
    models = load_all_models()
    
    print("\n[4] Finding common errors...")
    error_indices, all_predictions, all_probabilities = find_common_errors(models, X_test, y_test)
    
    if len(error_indices) == 0:
        print("\nNo common error samples found across all models!")
        return
    
    print("\n[5] Selecting representative samples...")
    selected_indices = select_samples(error_indices, all_probabilities, n_samples=2)
    print(f"Selected {len(selected_indices)} representative samples: {selected_indices}")
    
    print("\n[6] Visualizing samples...")
    for i, idx in enumerate(selected_indices, 1):
        visualize_sample(idx, X_test, y_test, all_predictions, all_probabilities, feature_names, i)
    
    print("\n[7] Creating summary plot...")
    create_summary_plot(selected_indices, X_test, y_test, all_predictions, all_probabilities, feature_names)
    
    print(f"Total common errors: {len(error_indices)}")
    print(f"Visualized samples: {len(selected_indices)}")


if __name__ == '__main__':
    main()
