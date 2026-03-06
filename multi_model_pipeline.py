import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    os.makedirs('plots', exist_ok=True)
    # Load dataset
    path = 'cardio_train.csv'
    df = pd.read_csv(path, sep=';')

    # cleaning
    df.drop('id', axis=1, inplace=True)
    df['age'] = df['age'] / 365.25  # Convert days to years
    df.drop_duplicates(inplace=True)

    # Feature Engineering: BMI
    # BMI = weight(kg) / (height(m))^2
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Filter unreasonable values (BP)
    # Systolic BP: 50-250, Diastolic BP: 30-150
    df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]

    # Split features and target
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models and parameters for GridSearch
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {'C': [0.1, 1, 10], 'max_iter': [1000]}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {'max_depth': [5, 10, 15], 'min_samples_split': [10, 50]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 15]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [5, 15], 'weights': ['uniform', 'distance']}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        }
    }

    # Store results
    results = {}

    for name, config in models.items():
        print(f"Training {name}...")
        
        # Grid Search with CV
        clf = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
        
        # Use scaled data for distance-based models
        if name in ['Logistic Regression', 'KNN']:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
        # Best params
        print(f"Best Params: {clf.best_params_}")
        
        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        results[name] = clf.best_estimator_
        
        # Confusion Matrix Visualization
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()

    # Feature Importance (Random Forest)
    rf_model = results['Random Forest']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()
