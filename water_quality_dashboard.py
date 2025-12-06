import streamlit as st
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Try to import optional dependencies with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Charts will be limited.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("Joblib not installed. Model loading disabled.")

# Page configuration
st.set_page_config(
    page_title="Water Quality Dashboard",
    page_icon="💧",
    layout="wide"
)

# Title
st.title("💧 Water Quality Monitoring Dashboard")
st.write("Real-time water quality monitoring and prediction")

# Initialize session state for storing predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Real-time Analysis", "Data Visualization", "Model Info"])
    
    st.markdown("---")
    st.write(f"Python version: {sys.version}")
    
    if not JOBLIB_AVAILABLE:
        st.error("⚠️ Joblib not installed")
        st.info("Add 'joblib' to requirements.txt")
    else:
        st.success("✅ Joblib available")
    
    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly not installed")
        st.info("Add 'plotly' to requirements.txt")

# Load model function with error handling
@st.cache_resource
def load_model():
    if not JOBLIB_AVAILABLE:
        return None, None, None
    
    try:
        # Try to load the model
        model = joblib.load('water_quality_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, label_encoder, feature_columns
    except FileNotFoundError as e:
        st.warning(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model
model, label_encoder, feature_columns = load_model()

if page == "Real-time Analysis":
    st.header("📊 Real-time Sensor Data Analysis")
    
    # Create input form
    with st.form("sensor_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", value=25.5)
            turbidity = st.number_input("Turbidity (NTU)", value=2.1)
            dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", value=7.5)
            
        with col2:
            ph = st.number_input("pH", value=7.2)
            ammonia = st.number_input("Ammonia (mg/L)", value=0.05)
            nitrate = st.number_input("Nitrate (mg/L)", value=10.0)
        
        submitted = st.form_submit_button("Analyze Water Quality")
        
        if submitted:
            # Store data
            sensor_data = {
                'timestamp': datetime.now(),
                'temperature': temperature,
                'turbidity': turbidity,
                'dissolved_oxygen': dissolved_oxygen,
                'ph': ph,
                'ammonia': ammonia,
                'nitrate': nitrate
            }
            
            # Rule-based analysis (fallback if model not available)
            quality_score = 0
            checks = []
            
            # pH check
            if 6.5 <= ph <= 8.5:
                quality_score += 1
                checks.append(("pH", "✅ Good", "Within optimal range (6.5-8.5)"))
            else:
                checks.append(("pH", "❌ Poor", f"Outside optimal range (current: {ph})"))
            
            # Temperature check
            if 20 <= temperature <= 30:
                quality_score += 1
                checks.append(("Temperature", "✅ Good", "Within optimal range (20-30°C)"))
            else:
                checks.append(("Temperature", "❌ Poor", f"Outside optimal range (current: {temperature}°C)"))
            
            # Dissolved oxygen check
            if dissolved_oxygen >= 6:
                quality_score += 1
                checks.append(("Dissolved Oxygen", "✅ Good", f"Sufficient ({dissolved_oxygen} mg/L)"))
            else:
                checks.append(("Dissolved Oxygen", "❌ Poor", f"Low ({dissolved_oxygen} mg/L)"))
            
            # Determine overall quality
            if quality_score >= 2:
                overall = "✅ Good"
                color = "green"
            elif quality_score >= 1:
                overall = "⚠️ Fair"
                color = "orange"
            else:
                overall = "❌ Poor"
                color = "red"
            
            # Display results
            st.markdown(f"### Overall Water Quality: <span style='color:{color}'>{overall}</span>", unsafe_allow_html=True)
            
            # Show individual checks
            st.subheader("Parameter Checks:")
            for param, status, detail in checks:
                st.write(f"**{param}:** {status} - {detail}")
            
            # Try model prediction if available
            if model is not None and label_encoder is not None:
                st.markdown("---")
                st.subheader("ML Model Prediction")
                try:
                    # Prepare data for model
                    # Note: You'll need to adjust this based on your actual feature columns
                    input_data = {
                        'temperature': temperature,
                        'turbidity': turbidity,
                        'dissolved_oxygen': dissolved_oxygen,
                        'ph': ph,
                        'ammonia_imputed_sklearn': ammonia,
                        'nitrate': nitrate,
                        # Add default values for other required features
                        'hour': datetime.now().hour,
                        'day_of_week': datetime.now().weekday(),
                        'month': datetime.now().month,
                        'year': datetime.now().year,
                        'population_imputed_sklearn': 1000,
                        'fish_weight': 500,
                        'fish_length_imputed_sklearn': 25
                    }
                    
                    # Create DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure all required columns are present
                    missing_cols = set(feature_columns) - set(input_df.columns)
                    for col in missing_cols:
                        input_df[col] = 0  # or appropriate default
                    
                    # Reorder columns
                    input_df = input_df[feature_columns]
                    
                    # Make prediction
                    prediction_encoded = model.predict(input_df)[0]
                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                    
                    st.success(f"Model Prediction: **{prediction}**")
                    
                except Exception as e:
                    st.error(f"Model prediction failed: {str(e)}")
            
            # Store in history
            st.session_state.predictions_history.append(sensor_data)

elif page == "Data Visualization":
    st.header("📈 Data Visualization")
    
    if st.session_state.predictions_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Display data
        st.subheader("Recent Measurements")
        st.dataframe(history_df)
        
        # Simple charts using Streamlit's built-in charts
        if 'temperature' in history_df.columns:
            st.subheader("Temperature Trend")
            st.line_chart(history_df['temperature'])
        
        if 'ph' in history_df.columns:
            st.subheader("pH Trend")
            st.line_chart(history_df['ph'])
        
        # Plotly charts if available
        if PLOTLY_AVAILABLE and len(history_df) > 1:
            st.subheader("Interactive Chart")
            fig = px.scatter(history_df, x='temperature', y='ph', 
                           color='dissolved_oxygen',
                           title='Temperature vs pH')
            st.plotly_chart(fig)
    else:
        st.info("No data available yet. Make some measurements first.")

elif page == "Model Info":
    st.header("🤖 Model Information")
    
    if model is not None:
        st.success("✅ Model successfully loaded!")
        
        # Show model info
        st.subheader("Model Details")
        st.write(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_estimators'):
            st.write(f"Number of trees: {model.n_estimators}")
        
        if feature_columns:
            st.write(f"Number of features: {len(feature_columns)}")
            st.write("Features used:", feature_columns)
    
    else:
        st.warning("⚠️ Model not loaded")
        st.info("""
        To enable ML predictions:
        1. Ensure model files are in the repository:
           - `water_quality_model.pkl`
           - `label_encoder.pkl`
           - `feature_columns.pkl`
        2. Check that joblib is in requirements.txt
        3. The model should be trained with scikit-learn 1.3.2
        """)
        
        # Show file structure
        st.subheader("Required Files")
        st.code("""
        your-repo/
        ├── water_quality_dashboard.py
        ├── requirements.txt
        ├── runtime.txt
        ├── water_quality_model.pkl
        ├── label_encoder.pkl
        └── feature_columns.pkl
        """)

# Footer
st.markdown("---")
st.markdown("💧 **Water Quality Monitoring System** | Made with Streamlit")
