# 🚀 VIP Customer Churn Predictor

This project is a two-stage machine learning pipeline designed to identify and predict churn for high-value "VIP" customers. The entire model is deployed as an interactive web application using Streamlit.



### ## 🎯 The Problem
Standard customer churn models are often inefficient. They treat all customers equally, sending the same "at-risk" alert for a low-value, one-time buyer as they do for a high-value "VIP."

Losing a VIP customer is far more costly. This project builds a "smarter" model that specifically answers the real business question: **"Which of our *most valuable* customers are at the *highest risk* of leaving?"**

---

### ## 🛠️ Technology & Tools
* **Python 3.11**
* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`LinearRegression`, `LogisticRegression`, `StandardScaler`, `PolynomialFeatures`)
* **Deployment:** Streamlit
* **Model Saving:** Joblib
* **Training Environment:** Google Colab

---

### ## ⚙️ Methodology: A Two-Stage Pipeline

This project uses a unique two-stage modeling approach to first quantify "value" and then predict "risk."

#### Method 1: The "Value" Model (Regression)
1.  **Objective:** To predict a customer's future financial worth.
2.  **Data:** The [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) was cleaned and engineered into **Recency (R), Frequency (F), and Monetary (M)** features.
3.  **Model:** A `Polynomial Linear Regression` model was trained on the RFM data to predict the `CLV_3_Month` (3-Month Customer Lifetime Value).
4.  **Output:** This model generates a "VIP Score" (`Predicted_CLV`) for each customer.

#### Method 2: The "Risk" Model (Classification)
1.  **Objective:** To predict the likelihood of a customer churning.
2.  **Model:** A `Logistic Regression` classifier.
3.  **Key Feature:** The model was trained on **RFM data *plus* the "VIP Score"** from Method 1. This "smart feature" allows the model to understand the *value* of the customer it's analyzing.

---

### ## 📈 Key Results
* **Regression Model (R-squared): 0.62
