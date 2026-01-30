#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Housing Price Prediction",
    layout="wide"
)

st.title("🏠 Housing Price Prediction App")

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Transformed_Housing_Data2.csv")

data = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(data.head())

# -------------------------------------------------
# Feature & Target
# -------------------------------------------------
Y = data["Sale_Price"]
X = data.drop(columns=["Sale_Price"])

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------------------------------
# VIF Calculation
# -------------------------------------------------
def calculate_vif(df):
    return pd.Series(
        [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
        index=df.columns
    )

vif = calculate_vif(X_scaled)
high_vif_features = vif[vif > 5].index.tolist()
X_vif = X_scaled.drop(columns=high_vif_features)

st.subheader("📉 Features after VIF Removal")
st.write(f"Removed features (VIF > 5): {high_vif_features}")
st.dataframe(X_vif.head())

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vif, Y, test_size=0.3, random_state=101
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)

st.subheader("📊 Model Performance")
st.metric("R² Score", f"{r2_score:.4f}")

# -------------------------------------------------
# Residual Plot
# -------------------------------------------------
predictions = model.predict(X_test)
residuals = predictions - y_test

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(predictions, residuals, s=10)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted Sale Price")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")

st.pyplot(fig)

# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("🔮 Predict Sale Price")

user_input = {}
for col in X_vif.columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict Price"):
    user_df = pd.DataFrame([user_input])
    user_scaled = pd.DataFrame(
        scaler.transform(user_df),
        columns=X_vif.columns
    )
    prediction = model.predict(user_scaled)
    st.success(f"💰 Predicted Sale Price: {prediction[0]:,.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




