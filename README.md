
# Cardiovascular Disease Risk Prediction

## Project Overview

This project aims to leverage machine learning techniques to classify and predict cardiovascular disease risk based on patients' clinical data. By analyzing multidimensional features—including objective facts, physical examination results, and subjective patient-reported information—we seek to build a high-accuracy predictive model to support early risk assessment and preventive medical interventions.

## Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of death worldwide. Accurate early risk assessment is critical for effective prevention and treatment planning. Existing risk assessment methods may not fully exploit the rich, multi-source clinical data now available. This project utilizes a publicly available dataset containing three categories of features:

- **Objective information**: factual data such as age and gender.  
- **Examination results**: clinical measurements like blood pressure and cholesterol levels.  
- **Subjective information**: self-reported symptoms or perceptions from patients.

Our goal is to develop a binary classification model that integrates these diverse inputs to predict whether an individual has cardiovascular disease, thereby providing data-driven support for clinical decision-making.

## Dataset

We use the publicly available dataset from Kaggle:  
[https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).  

The dataset contains approximately 70,000 samples, each with 11 features and 1 target variable indicating the presence or absence of cardiovascular disease.

## Features

### Core Machine Learning Pipeline
- **5 Classification Models**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
- **Comprehensive Evaluation**: Accuracy, F1-Score, Precision, Recall, AUC-ROC
- **Enhanced ROC Analysis**: Bootstrap confidence intervals for AUC curves
- **Feature Importance**: Random Forest-based feature ranking

### Model Interpretability
- **SHAP Value Analysis**: Model-agnostic feature importance
- **Dependence Plots**: Feature-prediction relationships
- **Summary Visualizations**: Beeswarm and bar plots

### Interactive Web Application
- **Streamlit Interface**: User-friendly prediction interface
- **Multi-Model Comparison**: Side-by-side predictions from all models
- **Confidence Visualization**: Probability distributions and metrics
- **Real-time Prediction**: Instant results with visual feedback

## Technologies and Tools

This project primarily uses the **Python** programming language along with the following libraries:

- **Data Processing**: `NumPy`, `Pandas`  
- **Data Visualization**: `Matplotlib`, `Seaborn`  
- **Machine Learning**: `Scikit-learn`
- **Model Interpretability**: `SHAP`
- **Web Application**: `Streamlit`
- **Testing**: `pytest`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Perform exploratory data analysis
- Train all 5 models with hyperparameter tuning
- Generate evaluation plots (ROC curves, confusion matrices, etc.)
- Perform SHAP value analysis
- Save trained models to `models/` directory

### 2. Run Interactive Web App

```bash
streamlit run app.py
```

This launches a web interface where you can:
- Input patient parameters (age, BMI, blood pressure, lifestyle factors)
- Get predictions from all trained models
- View confidence scores and probability distributions
- Compare model predictions side-by-side

### 3. Run Tests

```bash
pytest
```

Executes the test suite to verify code correctness.

## Project Structure

```
disease-analysis/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data cleaning and feature engineering
│   ├── exploration.py           # Exploratory data analysis
│   ├── models.py                # Model training and evaluation
│   └── shap_analysis.py         # SHAP value interpretation
├── tests/                        # Unit tests
│   ├── test_preprocessing.py
│   └── test_models.py
│   └── test_shap_analysis.py
├── plots/                        # Generated visualizations
│   ├── roc_curve_comparison.png # Enhanced ROC with confidence intervals
│   ├── confusion_matrix_*.png   # Per-model confusion matrices
│   ├── shap_summary.png         # SHAP value analysis
│   └── ...
├── models/                       # Saved trained models
│   ├── Logistic_Regression.joblib
│   ├── KNN.joblib
│   ├── Decision_Tree.joblib
│   ├── Random_Forest.joblib
│   └── Gradient_Boosting.joblib
├── results/                      # Analysis results
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── shap_importance.csv
│   └── data_summary.txt
├── app.py                        # Streamlit web application
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── cardio_train.csv              # Raw dataset
└── README.md                     # This file
```

## Model Performance

Models are evaluated using multiple metrics:
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balance between precision and recall
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive detection rate
- **AUC-ROC**: Area under receiver operating characteristic curve

The best performing model is automatically selected and used for SHAP analysis.

## Key Improvements

### Enhanced AUC Analysis
- Bootstrap confidence intervals (100 iterations)
- Shaded regions showing uncertainty bounds
- Improved visual clarity with color-coded models

### SHAP Interpretability
- Tree-based SHAP explainer for ensemble models
- Feature importance ranking
- Dependence plots showing feature impact
- Summary visualizations for global interpretation

### Code Quality
- Comprehensive error handling and input validation
- Unit tests covering edge cases
- Modular architecture with clear separation of concerns
- Model persistence with joblib
- Reproducible random seeds

### Web Application
- Interactive Streamlit interface
- Real-time prediction with multiple models
- Confidence score visualization
- Responsive design with input validation

## Team Responsibilities

- **[songbo]**: Responsible for data collection and cleaning, machine learning pipeline implementation, model training, and experimental evaluation.  
- **[keyan]**: Responsible for data visualization, literature review, and final report writing and integration.

## Disclaimer

This project is for educational purposes only. The predictions should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.
