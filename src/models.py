import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, precision_score, recall_score)

def train_and_evaluate(df):
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
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Store results
        all_results.append({
            'Model': name,
            'Accuracy': acc,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Best_Params': str(grid.best_params_)
        })
        
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
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
    
    print(results_df[['Model', 'Accuracy', 'F1_Score', 'Precision', 'Recall']].to_string(index=False))
    
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
    
    # Feature importance
    
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
    
    return results_df, best_model_name