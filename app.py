import streamlit as st
import joblib
import pandas as pd
import numpy as np


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


random_forest_regressor_model = joblib.load("random_forest_regressor_model.joblib")
label_encoders_dict = joblib.load("label_encoders.joblib")
median_imputation_values = joblib.load("median_imputation_values.joblib")
feature_names = joblib.load("feature_names.joblib")


data = pd.read_csv(r"car details v4.csv")

st.title(" Car Price Prediction App")
st.markdown("Enter the main car details to get an instant price estimate.")
st.divider()

# Important Fields Only
col1, col2 = st.columns(2)

with col1:
    selected_make = st.selectbox("Make", sorted(data['Make'].unique()))
    year = st.number_input("Year", min_value=1980, max_value=2023, value=2017)
    kilometer = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000)
    engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200)

with col2:
    filtered_models = sorted(data[data['Make'] == selected_make]['Model'].unique())
    model_name = st.selectbox("Model", filtered_models)
    fuel_type = st.selectbox("Fuel Type", label_encoders_dict['Fuel Type'].classes_)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=500, value=80)


# Hidden / Default Fields
location = label_encoders_dict['Location'].classes_[0]
color = label_encoders_dict['Color'].classes_[0]
owner = label_encoders_dict['Owner'].classes_[0]
seller_type = label_encoders_dict['Seller Type'].classes_[0]
drivetrain = label_encoders_dict['Drivetrain'].classes_[0]

length = median_imputation_values['Length']
width = median_imputation_values['Width']
height = median_imputation_values['Height']
seating_capacity = 5
fuel_tank_capacity = median_imputation_values['Fuel Tank Capacity']
max_torque = median_imputation_values['Max Torque']

st.divider()


# Prediction
if st.button("Predict Car Price", use_container_width=True):

    user_input_dict = {
        'Make': selected_make,
        'Model': model_name,
        'Year': year,
        'Kilometer': kilometer,
        'Fuel Type': fuel_type,
        'Location': location,
        'Color': color,
        'Owner': owner,
        'Seller Type': seller_type,
        'Engine': engine,
        'Max Power': max_power,
        'Max Torque': max_torque,
        'Drivetrain': drivetrain,
        'Length': length,
        'Width': width,
        'Height': height,
        'Seating Capacity': seating_capacity,
        'Fuel Tank Capacity': fuel_tank_capacity,
        'Transmission_Manual': 1 if transmission == 'Manual' else 0
    }

    input_df = pd.DataFrame([user_input_dict])

    # Encoding
    categorical_cols = ['Make','Model','Fuel Type','Location','Color','Owner','Drivetrain','Seller Type']

    for col in categorical_cols:
        try:
            input_df[col] = label_encoders_dict[col].transform(input_df[col])
        except:
            input_df[col] = 0

    # Numerical Imputation
    numerical_columns = ['Engine','Max Power','Max Torque','Length','Height','Width','Fuel Tank Capacity']

    for col in numerical_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(median_imputation_values[col])


    # Column Order
    input_df = input_df[feature_names]


    # Predict
    predicted_log_price = random_forest_regressor_model.predict(input_df)[0]
    predicted_price = np.exp(predicted_log_price)

    usd_price = predicted_price / 83

    st.success(f"💰 Estimated Car Price: **${usd_price:,.2f}**")
    st.metric("Price in USD", f"${usd_price:,.2f}")

