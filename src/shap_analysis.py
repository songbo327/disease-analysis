import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 15
plt.rcParams['figure.facecolor'] = '#FFFFFF'
import shap
import joblib


def analyze_shap_values(df, model_path='models/Gradient_Boosting.joblib'):
    """Analyze SHAP values for model interpretability"""
    print("\n[6] Analyzing SHAP Values...")
    
    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    use_scaled = model_data['use_scaled']
    
    # Prepare data
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    
    if use_scaled and scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X.values
    
    # Sample for faster computation
    sample_size = min(1000, len(X))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X_processed[sample_idx]
    
    # Create SHAP explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get SHAP values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig('plots/shap_summary.png', bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type='bar', show=False, plot_size=None)
        plt.tight_layout()
        plt.savefig('plots/shap_bar.png', bbox_inches='tight')
        plt.close()
        
        # Dependence plots for top features
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(shap_importance)[::-1][:5]
        
        for idx in top_indices[:3]:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values, X_sample, 
                               feature_names=feature_names,
                               show=False)
            plt.tight_layout()
            plt.savefig(f'plots/shap_dependence_{feature_names[idx]}.png', 
                       bbox_inches='tight')
            plt.close()
        
        # Save SHAP values
        shap_df = {
            'Feature': feature_names,
            'Mean_SHAP': np.abs(shap_values).mean(axis=0),
            'Std_SHAP': np.abs(shap_values).std(axis=0)
        }
        import pandas as pd
        shap_df = pd.DataFrame(shap_df).sort_values('Mean_SHAP', ascending=False)
        shap_df.to_csv('results/shap_importance.csv', index=False)
        
        print("SHAP analysis completed. Plots saved to plots/")
        
        return shap_values, X_sample, feature_names
    
    return None, None, None
