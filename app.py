import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("model.pkl", "rb"))

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide"
)

# ======================
# UI DESIGN
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #fdfbfb, #ebedee);
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #6a1b9a;
}

.sub {
    text-align: center;
    font-size: 18px;
    color: #444;
}

.stButton button {
    background-color: #6a1b9a;
    color: white;
    border-radius: 10px;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown('<div class="title">🌸 Iris Flower Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Machine Learning Powered Classification App</div>', unsafe_allow_html=True)
st.markdown("---")

# ======================
# MENU
# ======================
menu = st.sidebar.selectbox("📌 Menu", ["Prediction", "Analytics", "Dataset Info"])

# ======================
# PAGE 1 — PREDICTION
# ======================
if menu == "Prediction":

    st.subheader("📏 Enter Flower Measurements")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0)

    with col2:
        petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 1.0)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("🌸 Predict Flower Type", use_container_width=True):

        prediction = model.predict(input_data)[0]

        classes = ["Setosa", "Versicolor", "Virginica"]

        st.markdown("### 🧾 Prediction Result")

        st.success(f"🌸 Flower Type: **{classes[prediction]}**")

        # probability (if supported)
        try:
            proba = model.predict_proba(input_data)[0]
            st.write("### 📊 Probability Distribution")
            st.bar_chart(proba)
        except:
            pass

# ======================
# PAGE 2 — ANALYTICS
# ======================
elif menu == "Analytics":

    st.subheader("📊 Feature Importance (Simulated)")

    df = pd.DataFrame({
        "Feature": ["Petal Length", "Petal Width", "Sepal Length", "Sepal Width"],
        "Importance": [0.45, 0.30, 0.15, 0.10]
    })

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=df, ax=ax)
    st.pyplot(fig)

    st.info("📌 Petal features are most important for classification.")

# ======================
# PAGE 3 — DATASET INFO
# ======================
elif menu == "Dataset Info":

    st.subheader("🌼 About Iris Dataset")

    st.write("""
    The Iris dataset is a classic ML dataset used for classification.

    ✔ 3 classes:
    - Setosa
    - Versicolor
    - Virginica

    ✔ Features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width
    """)

    st.success("Beginner-friendly ML classification project 🚀")