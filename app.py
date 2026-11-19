from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "Bengaluru_House_Data.csv"

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below. The trained Random Forest model "
    "will estimate the house price using the supplied dataset."
)

if not MODEL_PATH.exists():
    st.warning("The trained model has not been created yet. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

if DATA_PATH.exists():
    source = pd.read_csv(DATA_PATH)
    locations = sorted(source["location"].dropna().astype(str).str.strip().unique())
else:
    locations = ["Whitefield", "Sarjapur Road", "Electronic City", "Marathahalli"]

with st.form("prediction_form"):
    location = st.selectbox("Location", locations)
    area = st.number_input("House area (square feet)", min_value=100.0, max_value=20000.0, value=1500.0, step=50.0)
    bedrooms = st.number_input("Number of bedrooms (BHK)", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=2, step=1)
    submitted = st.form_submit_button("Predict House Price")

if submitted:
    house = pd.DataFrame([{
        "location": location,
        "total_sqft": area,
        "bath": bathrooms,
        "bhk": bedrooms
    }])
    prediction = float(model.predict(house)[0])
    st.success(f"Estimated house price: ₹{prediction:,.2f} lakhs")
    st.caption("This is an educational estimate, not a professional property valuation.")
