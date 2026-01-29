#!/usr/bin/env python
# coding: utf-8

# In[4]:




# In[6]:


get_ipython().run_cell_magic('writefile', 'app.py', '\nimport streamlit as st\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom statsmodels.stats.outliers_influence import variance_inflation_factor\nimport warnings\nwarnings.filterwarnings("ignore")\n\nst.set_page_config(page_title="Housing Price Prediction", layout="wide")\n\nst.title("🏠 Housing Price Prediction using Linear Regression")\nst.write("This app implements scaling, multicollinearity removal, linear regression, and evaluation.")\n\nst.sidebar.header("📂 Dataset")\nuploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])\n\nif uploaded_file is not None:\n    data = pd.read_csv(uploaded_file)\n    st.subheader("Raw Dataset")\n    st.dataframe(data.head())\n\n    st.subheader("🔄 Feature Scaling")\n    scaler = StandardScaler()\n    Y = data[\'Sale_Price\']\n    X = data.drop(columns=[\'Sale_Price\'])\n    X_scaled = scaler.fit_transform(X)\n    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)\n    st.dataframe(X_scaled.head())\n\n    st.subheader("🧮 Multicollinearity Check (VIF)")\n\n    def calculate_vif(df):\n        return pd.Series(\n            [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],\n            index=df.columns\n        )\n\n    vif_data = X_scaled.copy()\n    for _ in range(7):\n        vif = calculate_vif(vif_data)\n        if vif.max() > 5:\n            vif_data = vif_data.drop(columns=[vif.idxmax()])\n        else:\n            break\n\n    st.write("Remaining Features:", vif_data.columns.tolist())\n    st.dataframe(calculate_vif(vif_data))\n\n    st.subheader("✂ Train-Test Split")\n    X_train, X_test, y_train, y_test = train_test_split(\n        vif_data, Y, test_size=0.3, random_state=101\n    )\n\n    st.subheader("📈 Linear Regression Model")\n    model = LinearRegression()\n    model.fit(X_train, y_train)\n    st.success(f"R² Score: {model.score(X_test, y_test):.4f}")\n\n    predictions = model.predict(X_test)\n    residuals = predictions - y_test\n\n    st.subheader("📉 Residual Plot")\n    fig, ax = plt.subplots()\n    ax.scatter(predictions, residuals, s=5)\n    ax.axhline(0)\n    st.pyplot(fig)\n\nelse:\n    st.info("👈 Upload `Transformed_Housing_Data2.csv` to start")\n')


# In[ ]:





# In[ ]:





# In[ ]:




