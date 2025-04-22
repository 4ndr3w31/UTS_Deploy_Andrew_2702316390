import streamlit as st
import pandas as pd
from model_rf import predict_new  # Importing the prediction function

st.title("🎯 Prediksi Booking Status Hotel")

# Input features
st.header("Input Fitur:")
feature_names = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
    'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
    'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
    'arrival_date', 'market_segment_type', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
    'avg_price_per_room', 'no_of_special_requests'
]

# Dictionary to store user inputs
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"{feature}", step=1.0)

# When the button is clicked
if st.button("Prediksi"):
    input_df = pd.DataFrame([inputs])
    
    # Call the predict_new function to get the prediction
    result = predict_new(input_df)
    
    if result is not None:
        st.success(f"Booking Status: {'✅ Not Canceled' if result[0] == 0 else '❌ Canceled'}")
    else:
        st.error("Error during prediction.")
