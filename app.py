import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

# Title and divider
st.set_page_config(page_title="Salary Estimation App", layout="wide")
st.title("💰 Salary Estimation App")
st.markdown("### Predict your expected salary based on company experience!")

# Cute animated gif
st.image(
    "https://tenor.com/view/vrbie-vrbs-nft-nouns-farel-gif-9010198357377832888.gif", 
    caption="Let's predict!", 
    use_container_width=True
)

# Divider
st.divider()

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input("🔢 Years at company", min_value=0, max_value=20, value=3)

with col2:
    satisfaction_level = st.slider("😊 Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01, value=0.7)

with col3:
    average_monthly_hours = st.slider("🕓 Avg Monthly Hours", min_value=120, max_value=310, step=1, value=160)
                                      
# Collect input features into a list
X = [years_at_company, satisfaction_level, average_monthly_hours]

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Predict button
predict_button = st.button("🧠 Predict Salary")

st.divider()

if predict_button:
    st.balloons()

    # Prepare data for prediction
    X_array = scaler.transform([np.array(X)])
    prediction = model.predict(X_array)

    # Display the prediction
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")

    # Visualize user input
    df_input = pd.DataFrame({
        "Feature": ["Years at Company", "Satisfaction Level", "Average Monthly Hours"],
        "Value": X
    })

    fig = px.bar(df_input, x="Feature", y="Value", color="Feature", 
                 title="📊 Your Input Profile", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)