import streamlit as st
import pandas as pd
import pickle as pkl
# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💰",
    layout="centered"
)

# -----------------------------------
# Load trained model (cached)
# -----------------------------------
import pickle as pkl

@st.cache_data
def load_model():
    with open("insurance_premium_model.pkl", "rb") as f:
        model = pkl.load(f)
    return model

model = load_model()

# -----------------------------------
# App title & description
# -----------------------------------
st.title("💰 Insurance Premium Predictor")
st.markdown(
    """
    Predict your **insurance premium** using a machine learning model  
    built with **Random Forest Regression**.
    """
)

st.divider()

# -----------------------------------
# User input section
# -----------------------------------
st.subheader("Enter Customer Details")

age = st.slider("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", ["male", "female"])

bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0)

children = st.slider("Number of Children", min_value=0, max_value=5, value=0)

smoker = st.selectbox("Smoker", ["No", "Yes"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# -----------------------------------
# Encode inputs (SAME as training)
# -----------------------------------
gender_encoded = 1 if gender == "male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

region_map = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region_encoded = region_map[region]

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔮 Predict Premium"):
    input_df = pd.DataFrame(
        [[
            age,
            gender_encoded,
            bmi,
            children,
            smoker_encoded,
            region_encoded
        ]],
        columns=[
            "age",
            "gender",
            "bmi",
            "children",
            "smoker",
            "region"
        ]
    )

    prediction = model.predict(input_df)[0]

    st.success(f"💵 Estimated Insurance Premium: ₹ {prediction:,.2f}")

# -----------------------------------
# Footer
# -----------------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit & Machine Learning")
