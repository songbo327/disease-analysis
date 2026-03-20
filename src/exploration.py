import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    # Dataset statistics
    print(df.describe())
    
    # Class distribution
    class_counts = df['cardio'].value_counts()
    print(class_counts)
    print(f"Class balance ratio: {class_counts[0]/class_counts[1]:.2f}")
    
    with open('results/data_summary.txt', 'w') as f:
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {df.shape[1] - 1}\n")
        f.write(f"Target: cardio (binary)\n\n")
        f.write("Class Distribution:\n")
        f.write(f"  No disease (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)\n")
        f.write(f"  Disease (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)\n\n")
        f.write("Feature Statistics:\n")
        f.write(df.describe().to_string())
    
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    df['cardio'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Class Distribution')
    plt.xlabel('Cardiovascular Disease')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No Disease', 'Disease'], rotation=0)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    plt.close()
    
    # Plot feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 
                'gluc', 'smoke', 'alco', 'active', 'bmi', 'gender']
    for idx, feat in enumerate(features):
        ax = axes[idx // 4, idx % 4]
        df[feat].hist(ax=ax, bins=30, edgecolor='black')
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig('plots/feature_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()