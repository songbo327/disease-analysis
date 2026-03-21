
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

## Team Responsibilities

- **[songbo]**: Responsible for data collection and cleaning, machine learning pipeline implementation, model training, and experimental evaluation.  
- **[keyan]**: Responsible for data visualization, literature review, and final report writing and integration.

## Technologies and Tools

This project primarily uses the **Python** programming language along with the following libraries:

- **Data Processing**: `NumPy`, `Pandas`  
- **Data Visualization**: `Matplotlib`, `Seaborn`  
- **Machine Learning**: `Scikit-learn`

The focus of this project is to reinforce and deepen our existing skills within the Python data science ecosystem.

## Expected Project Structure

```
project-repo/
├── src/                     # Source code modules 
├── reports/                 # Generated reports and figures
├── README.md                # This file
├── cardio_train.csv         # Raw dataset
└── report.pdf               # Project report
└── download_dataset.py      # Script to download the dataset
```
