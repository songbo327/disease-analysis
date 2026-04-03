import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob


st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("Cardiovascular Disease Risk Prediction")
st.markdown("Input patient parameters to predict cardiovascular disease risk using multiple ML models.")


@st.cache_resource
def load_all_models():
    """Load all trained models"""
    models = {}
    model_files = glob.glob('models/*.joblib')
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.joblib', '')
        try:
            model_data = joblib.load(model_file)
            models[model_name] = model_data
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
    
    return models


def validate_input(name, value, min_val, max_val):
    """Validate user input"""
    if value < min_val or value > max_val:
        st.error(f"{name} must be between {min_val} and {max_val}")
        return False
    return True


def make_prediction(models, input_data):
    """Make predictions using all loaded models"""
    results = []
    
    for model_name, model_data in models.items():
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            use_scaled = model_data['use_scaled']
            
            # Create feature vector in correct order
            features = []
            for feat in feature_names:
                if feat in input_data:
                    features.append(input_data[feat])
                else:
                    features.append(0)
            
            X = np.array([features])
            
            # Scale if needed
            if use_scaled and scaler is not None:
                X = scaler.transform(X)
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            results.append({
                'Model': model_name.replace('_', ' '),
                'Prediction': 'Disease' if prediction == 1 else 'No Disease',
                'Confidence': f"{max(probability) * 100:.1f}%",
                'Probability_Disease': probability[1] * 100,
                'Probability_No_Disease': probability[0] * 100
            })
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")
    
    return pd.DataFrame(results)


# Load models
with st.spinner("Loading models..."):
    models = load_all_models()

if not models:
    st.warning("No trained models found. Please run main.py first to train models.")
    st.stop()

st.success(f"✓ Loaded {len(models)} models successfully")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
    gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 2 else "Female")
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    st.info(f"Calculated BMI: {bmi:.1f}")

with col2:
    st.subheader("Clinical Measurements")
    
    ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
    ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=130, value=80, step=1)
    cholesterol = st.selectbox("Cholesterol Level", 
                               options=[1, 2, 3],
                               format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    glucose = st.selectbox("Glucose Level",
                          options=[1, 2, 3],
                          format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])

st.subheader("Lifestyle Factors")
col3, col4, col5 = st.columns(3)

with col3:
    smoke = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col4:
    alco = st.selectbox("Alcohol Intake", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col5:
    active = st.selectbox("Physical Activity", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Prepare input data
input_data = {
    'age': age * 365.25,
    'gender': gender,
    'height': height,
    'weight': weight,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': glucose,
    'smoke': smoke,
    'alco': alco,
    'active': active,
    'bmi': bmi
}

# Predict button
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    with st.spinner("Making predictions..."):
        results_df = make_prediction(models, input_data)
    
    if not results_df.empty:
        st.subheader("Prediction Results")
        
        # Display results as cards
        cols = st.columns(len(results_df))
        for idx, (_, row) in enumerate(results_df.iterrows()):
            with cols[idx]:
                st.metric(
                    label=row['Model'],
                    value=row['Prediction'],
                    delta=f"Confidence: {row['Confidence']}"
                )
        
        # Detailed results table
        st.subheader("Detailed Results")
        display_df = results_df[['Model', 'Prediction', 'Confidence']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Probability visualization
        st.subheader("Risk Probability Comparison")
        
        prob_df = results_df[['Model', 'Probability_Disease', 'Probability_No_Disease']].copy()
        prob_df = prob_df.set_index('Model')
        
        st.bar_chart(prob_df, use_container_width=True)
        
        # Best model recommendation
        best_model = results_df.loc[results_df['Probability_Disease'].idxmax()]
        st.info(f"**Highest Risk Detection**: {best_model['Model']} predicts {best_model['Prediction']} "
               f"with {best_model['Confidence']} confidence")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses machine learning models trained on cardiovascular disease data to predict risk.
    
    **Models Available:**
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    """)
