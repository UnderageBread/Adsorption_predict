import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="PFAS Adsorption Prediction", layout="wide")

type_dict = {
    'Temperature': 'Biochar production conditions',
    'Pyrolysis time': 'Biochar production conditions',
    'Activation/modification': 'Biochar production conditions',
    'Surface area': 'Biochar properties',
    'Pore volume': 'Biochar properties',
    'Average pore size': 'Biochar properties',
    'C': 'Biochar properties',
    'O': 'Biochar properties',
    'N': 'Biochar properties',
    'H': 'Biochar properties',
    'O/C': 'Biochar properties',
    'N/C': 'Biochar properties',
    '(O+N)/C': 'Biochar properties',
    'H/C': 'Biochar properties',
    'Ash': 'Biochar properties',
    'Fe': 'Biochar properties',
    'Adsorption time': 'Adsorption conditions',
    'Initial PFAS concentration': 'Adsorption conditions',
    'Solution pH': 'Adsorption conditions',
    'BC concentration ': 'Adsorption conditions',
    'RPM': 'Adsorption conditions',
    'Adsorption temperature': 'Adsorption conditions',
    'Molar mass ': 'PFAS properties',
    'F/C': 'PFAS properties',
    'Chain length': 'PFAS properties',
    'Head group': 'PFAS properties'
}

unit_dict = {
    'Temperature': '°C',
    'Pyrolysis time': 'h',
    'Activation/modification': '',
    'Surface area': 'm²/g',
    'Pore volume': 'cm³/g',
    'Average pore size': 'nm',
    'C': '%',
    'O': '%',
    'N': '%',
    'H': '%',
    'O/C': 'mol/mol',
    'N/C': 'mol/mol',
    '(O+N)/C': 'mol/mol',
    'H/C': 'mol/mol',
    'Ash': '%',
    'Fe': '%',
    'Adsorption time': 'h',
    'Initial PFAS concentration': 'mg/L',
    'Solution pH': '',
    'BC concentration ': 'mg/L',
    'RPM': '',
    'Adsorption temperature': '°C',
    'Molar mass ': 'g/mol',
    'F/C': 'mol/mol',
    'Chain length': '',
    'Head group': ''
}

@st.cache_resource
def load_models_and_data():
    with open('feature_info.pkl', 'rb') as f:
        feature_info = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    with open('models_trained.pkl', 'rb') as f:
        models = pickle.load(f)
    
    data = pd.read_excel("PFAS数据.xlsx", header=4, index_col=0)
    
    return feature_info, encoders, scaler_X, scaler_y, models, data

feature_info, encoders, scaler_X, scaler_y, models, data = load_models_and_data()

st.title("PFAS Adsorption Capacity Prediction")
st.markdown("---")

use_defaults = st.checkbox("Use default values from first sample", value=True)

features = feature_info['all_features']
cat_features = feature_info['categorical_features']
num_features = feature_info['numerical_features']

first_sample = data.iloc[0]

input_data = {}

feature_categories = [
    'Biochar production conditions',
    'Biochar properties',
    'Adsorption conditions',
    'PFAS properties'
]

for category in feature_categories:
    st.subheader(category)
    
    category_features = [f for f in features if type_dict.get(f) == category]
    
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(category_features):
        col = col1 if idx % 2 == 0 else col2
        
        unit = unit_dict.get(feature, '')
        feature_label = f"{feature} ({unit})" if unit else feature
        
        with col:
            if feature in cat_features:
                categories = encoders[feature].classes_.tolist()
                
                if use_defaults and feature in first_sample.index and pd.notna(first_sample[feature]):
                    default_value = first_sample[feature]
                    if default_value in categories:
                        default_idx = categories.index(default_value)
                    else:
                        default_idx = 0
                else:
                    default_idx = 0
                
                selected_category = st.selectbox(
                    feature_label,
                    options=categories,
                    index=default_idx,
                    key=feature
                )
                
                input_data[feature] = encoders[feature].transform([selected_category])[0]
                
            else:
                if use_defaults and feature in first_sample.index and pd.notna(first_sample[feature]):
                    default_value = float(first_sample[feature])
                else:
                    default_value = 0.0
                
                value = st.number_input(
                    feature_label,
                    value=default_value,
                    format="%.4f",
                    key=feature
                )
                
                input_data[feature] = value
    
    st.markdown("---")

st.markdown("---")

model_name = st.selectbox(
    "Select Model",
    options=list(models.keys())
)

if st.button("Predict", type="primary"):
    input_array = np.array([input_data[f] for f in features]).reshape(1, -1)
    
    input_scaled = scaler_X.transform(input_array)
    
    selected_model = models[model_name]
    prediction_scaled = selected_model.predict(input_scaled)
    
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    
    st.success(f"Predicted Adsorption Capacity: {prediction:.4f} mg/g")
    
    st.markdown("---")
    
    st.subheader("Input Summary")
    
    input_df = pd.DataFrame({
        'Feature': features,
        'Input Value': [input_data[f] for f in features]
    })
    
    st.dataframe(input_df, use_container_width=True)