
import streamlit as st
import numpy as np
import pandas
import sklearn
import pickle

# Load the model and scalers
try:
    rfc = pickle.load(open('model.pkl', 'rb'))
    SC = pickle.load(open('Standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
     
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure they are in the same directory as the script.")
    st.stop()  # Stop execution if files are missing


# Crop dictionary (same as before)
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Streamlit app
st.title("Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, value=0)
P = st.number_input("Phosphorus (P)", min_value=0, value=0)
K = st.number_input("Potassium (K)", min_value=0, value=0)
temp = st.number_input("Temperature (°C)", min_value=0.0, value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, value=0.0)
ph = st.number_input("pH", min_value=0.0, value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)


if st.button("Predict"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Handle potential errors during scaling
    try:
        scaled_features = ms.transform(single_pred)
        final_features = SC.transform(scaled_features)
        prediction = rfc.predict(final_features)

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there." # Use f-string for cleaner formatting
            st.success(result) # Display result in a success box
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            st.warning(result) # Display result in a warning box

    except ValueError as e: # Catch scaling errors (e.g. if user inputs are unexpected types)
        st.error(f"An error occurred during processing: {e}")
        st.error("Please check your input values.")
    except Exception as e:  # Catch any other potential errors
        st.error(f"An unexpected error occurred: {e}")