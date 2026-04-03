import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.exploration import explore_data
from src.models import train_and_evaluate

# Set seeds
random.seed(42)
np.random.seed(42)

# Create output dirs
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
df = load_data('cardio_train.csv')
print(f"Original shape: {df.shape}")

# Preprocess
df = preprocess_data(df)
print(f"After preprocessing: {df.shape}")

# Explore
explore_data(df)

# Train models
results_df, best_model_name, trained_models = train_and_evaluate(df, save_models=True)

# SHAP analysis
from src.shap_analysis import analyze_shap_values
if trained_models and best_model_name:
    model_path = f'models/{best_model_name}.joblib'
    if os.path.exists(model_path):
        analyze_shap_values(df, model_path)

# Final summary
print(f"\nBest Model: {best_model_name}")