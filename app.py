import streamlit as st
import joblib
import numpy as np

# --- 1. LOAD THE SAVED FILES ---

# Use st.cache_resource to load models only once
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('x_scaler.pkl')
        poly = joblib.load('poly_features.pkl')
        regressor = joblib.load('regression_model.pkl')
        classifier = joblib.load('classification_model.pkl')
        return scaler, poly, regressor, classifier
    except FileNotFoundError:
        st.error("Model files not found. Please make sure all .pkl files are in the same directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

scaler, poly, regressor, classifier = load_models()

# --- 2. SET UP THE STREAMLIT INTERFACE ---

st.set_page_config(page_title="VIP Churn Predictor", layout="centered")
st.title("🚀 VIP Customer Churn Predictor")
st.write("""
Enter a customer's RFM data to predict their churn risk.
This app uses a two-stage model:
1.  **Regression:** Predicts the customer's *potential value*.
2.  **Classification:** Predicts *churn risk* using RFM + potential value.
         
🗓️ Recency
What it means: How recently did this customer make a purchase?
What to input: The number of days since their last order.

🛒 Frequency
What it means: How often does this customer buy?
What to input: The total number of purchases (or orders) they have made in their entire history.

💰 Monetary
What it means: How much money has this customer spent?
What to input: The total sum of money (e.g., in dollars) they have spent across all their purchases.
""")

st.divider()

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (Days)", 
        min_value=0, 
        help="How many days ago was their last purchase?"
    )

with col2:
    frequency = st.number_input(
        "Frequency (Purchases)", 
        min_value=1, 
        help="How many total purchases have they made?"
    )

with col3:
    monetary = st.number_input(
        "Monetary (Total $)", 
        min_value=0.01, 
        format="%.2f", 
        help="What is their total lifetime spending?"
    )

# --- 3. THE PREDICTION LOGIC ---

# Check if models are loaded before showing the button
if scaler and poly and regressor and classifier:
    
    # Create the prediction button
    if st.button("Predict Churn Risk", type="primary"):
        
        # --- STAGE 1: REGRESSION PREDICTION ---
        
        # 1. Create the input array for scaling (must be 2D)
        input_data_regr = np.array([[recency, frequency, monetary]])
        
        # 2. Scale the input data
        input_scaled = scaler.transform(input_data_regr)
        
        # 3. Apply polynomial features
        input_poly = poly.transform(input_scaled)
        
        # 4. Predict the CLV (this is the scaled `Predicted_CLV`)
        predicted_clv_scaled = regressor.predict(input_poly)[0]

        
        # --- STAGE 2: CLASSIFICATION PREDICTION ---
        
        # 1. Create the feature array for the classifier
        #    (Raw R, F, M + the new scaled CLV prediction)
        input_data_class = np.array([[
            recency, 
            frequency, 
            monetary, 
            predicted_clv_scaled 
        ]])
        
        # 2. Predict the churn (0 or 1)
        prediction = classifier.predict(input_data_class)[0]
        
        # 3. Get the probability of churn (class 1)
        probability = classifier.predict_proba(input_data_class)[0][1]
        
        
        # --- 4. DISPLAY THE RESULT ---
        
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"🚨 **At Risk of Churn** (Probability: {probability*100:.2f}%)")
            st.warning("This is a high-value customer who is likely to leave. Recommend sending a retention offer.")
        else:
            st.success(f"✅ **Likely to Stay** (Churn Probability: {probability*100:.2f}%)")
            st.info("This customer is not a high churn risk. Standard marketing is fine.")

        # (Optional) Show the intermediate step
        with st.expander("See model details"):
            st.write(f"Intermediate `Predicted_CLV` (scaled): {predicted_clv_scaled:.4f}")
            st.write(f"Final features sent to classifier: {input_data_class.flatten()}")

else:
    st.warning("Models are not loaded. Please check your .pkl files.")