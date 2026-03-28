import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# 🎨 Header
st.title("🌸 Iris Flower Classification App")
st.write("Predict the species of Iris flower using ML model")

st.divider()

# 🌿 Input Section (clean UI)
st.subheader("📏 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

st.divider()

# 🔮 Prediction
if st.button("🌸 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa 🌼", "Versicolor 🌺", "Virginica 🌸"]

    st.success(f"Predicted Species: **{species[prediction[0]]}**")