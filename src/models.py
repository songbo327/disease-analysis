import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 15
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['figure.facecolor'] = '#FFFFFF'
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score,
                             roc_curve, auc, roc_auc_score)
import joblib
import os

def plot_enhanced_roc(roc_data, y_test):
    """Plot enhanced ROC curves with confidence intervals"""
    
    plt.figure(figsize=(12, 10))
    colors = {'Logistic_Regression': '#1f77b4', 'KNN': '#ff7f0e', 
              'Decision_Tree': '#2ca02c', 'Random_Forest': '#d62728', 
              'Gradient_Boosting': '#9467bd'}
    
    for name, data in roc_data.items():
        fpr = data['fpr']
        tpr = data['tpr']
        roc_auc = data['auc']
        
        # Calculate confidence intervals using bootstrap
        n_bootstraps = 100
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        np.random.seed(42)
        for i in range(n_bootstraps):
            indices = np.random.randint(0, len(y_test), len(y_test))
            if len(np.unique(y_test.iloc[indices])) < 2:
                continue
            
            y_prob_array = data['y_prob']
            fpr_boot, tpr_boot, _ = roc_curve(y_test.iloc[indices], y_prob_array[indices])
            tprs.append(np.interp(mean_fpr, fpr_boot, tpr_boot))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr_boot, tpr_boot))
        
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        # Plot mean ROC curve
        plt.plot(fpr, tpr, color=colors[name], linewidth=2.5,
                label=f'{name.replace("_", " ")} (AUC = {roc_auc:.3f})')
        
        # Plot confidence interval
        tprs_lower_interp = np.interp(fpr, mean_fpr, tprs_lower)
        tprs_upper_interp = np.interp(fpr, mean_fpr, tprs_upper)
        plt.fill_between(fpr, tprs_lower_interp, tprs_upper_interp, color=colors[name], alpha=0.15,
                        label='_nolegend_')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Enhanced ROC Curve Comparison with Confidence Intervals', fontsize=16)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve_comparison.png')
    plt.close()


def train_and_evaluate(df, save_models=True):
    """Train models and compare results"""
    # Split features and target
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(),
            'params': {'C': [0.1, 1.0, 10.0], 'max_iter': [1000]},
            'use_scaled': True
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance']},
            'use_scaled': True
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [5, 10, 15], 'min_samples_split': [10, 50, 100]},
            'use_scaled': False
        },
        'Random_Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 15, 20]},
            'use_scaled': False
        },
        'Gradient_Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
            'use_scaled': False
        }
    }
    
    # Train and evaluate
    all_results = []
    roc_data = {}
    trained_models = {}
    
    for name, config in models.items():
        
        if config['use_scaled']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Grid search
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_tr, y_train)
        
        # Predictions
        y_pred = grid.predict(X_te)
        y_prob = grid.predict_proba(X_te)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'y_prob': y_prob}
        
        # Store results
        all_results.append({
            'Model': name,
            'Accuracy': acc,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC': roc_auc,
            'Best_Params': str(grid.best_params_)
        })
        
        print(f"\n{name}")
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        # Save model
        if save_models:
            os.makedirs('models', exist_ok=True)
            model_data = {
                'model': grid.best_estimator_,
                'scaler': scaler if config['use_scaled'] else None,
                'feature_names': feature_names,
                'use_scaled': config['use_scaled']
            }
            joblib.dump(model_data, f'models/{name}.joblib')
            trained_models[name] = model_data
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{name}.png')
        plt.close()
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    results_df.to_csv('results/model_comparison.csv', index=False)
    
    print("Model Comparison Results")
    print(results_df[['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall', 'AUC']].to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.2
    
    plt.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', color='steelblue')
    plt.bar(x - 0.5*width, results_df['F1_Score'], width, label='F1 Score', color='darkorange')
    plt.bar(x + 0.5*width, results_df['Precision'], width, label='Precision', color='green')
    plt.bar(x + 1.5*width, results_df['Recall'], width, label='Recall', color='red')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Enhanced ROC curves with confidence intervals
    plot_enhanced_roc(roc_data, y_test)
    
    # Feature importance (Random Forest)
    print("\n[5] Analyzing Feature Importance...")
    
    best_rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42
    )
    best_rf.fit(X_train, y_train)
    
    importance = best_rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[indices], align='center')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    # 
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    feat_imp_df.to_csv('results/feature_importance.csv', index=False)
    
    
    best_model_name = results_df.iloc[0]['Model']
    
    return results_df, best_model_name, trained_models if save_models else None