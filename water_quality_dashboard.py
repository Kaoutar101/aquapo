import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Water Quality Monitoring Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
    }
    .prediction-card {
        background-color: #e6f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning {
        background-color: #fffacd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
    }
    .danger {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ff5252;
    }
    .safe {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and encoders (with error handling)
@st.cache_resource
def load_models():
    try:
        model = joblib.load('water_quality_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, label_encoder, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

# Initialize session state for storing predictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Title and description
st.markdown('<h1 class="main-header">💧 Water Quality Monitoring Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
    Real-time water quality prediction using Random Forest model. 
    This dashboard visualizes predictions based on sensor data from Arduino devices.
""")

# Load models
model, label_encoder, feature_columns = load_models()

# Sidebar for navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/water.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["Real-time Prediction", "Historical Data", "Model Performance", "About"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    if model is not None:
        st.metric("Model Accuracy", "95.2%")  # Update with your actual accuracy
        st.metric("Total Predictions", len(st.session_state.predictions_history))
    
    st.markdown("---")
    st.markdown("### 🎛️ Sensor Settings")
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
    
    st.markdown("---")
    st.markdown("Built with ❤️ for Water Quality Monitoring")

# Page 1: Real-time Prediction
if page == "Real-time Prediction":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📈 Real-time Sensor Data</h2>', unsafe_allow_html=True)
        
        # Create tabs for different input methods
        input_method = st.radio(
            "Select input method:",
            ["Manual Input", "Simulate Arduino Data", "Upload CSV"]
        )
        
        if input_method == "Manual Input":
            # Create input form for sensor data
            with st.form("sensor_data_form"):
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    temperature = st.number_input("Temperature (°C)", value=25.5, min_value=0.0, max_value=50.0, step=0.1)
                    turbidity = st.number_input("Turbidity (NTU)", value=2.1, min_value=0.0, max_value=100.0, step=0.1)
                    dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", value=7.5, min_value=0.0, max_value=20.0, step=0.1)
                    ph = st.number_input("pH", value=7.2, min_value=0.0, max_value=14.0, step=0.1)
                    
                with col1_2:
                    nitrate = st.number_input("Nitrate (mg/L)", value=10.0, min_value=0.0, max_value=100.0, step=0.1)
                    ammonia = st.number_input("Ammonia (mg/L)", value=0.05, min_value=0.0, max_value=10.0, step=0.01)
                    fish_weight = st.number_input("Fish Weight (g)", value=500.0, min_value=0.0, max_value=5000.0, step=1.0)
                    fish_length = st.number_input("Fish Length (cm)", value=25.0, min_value=0.0, max_value=100.0, step=0.1)
                
                # Time-based features
                current_time = datetime.now()
                hour = st.slider("Hour of Day", 0, 23, current_time.hour)
                month = st.slider("Month", 1, 12, current_time.month)
                year = st.slider("Year", 2020, 2030, 2024)
                
                # Additional features
                population = st.number_input("Population", value=1000.0, min_value=0.0, max_value=10000.0, step=10.0)
                day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, current_time.weekday())
                
                submit_button = st.form_submit_button("Predict Water Quality")
                
                if submit_button:
                    # Prepare sensor data
                    sensor_data = {
                        'temperature': temperature,
                        'turbidity': turbidity,
                        'dissolved_oxygen': dissolved_oxygen,
                        'ph': ph,
                        'nitrate': nitrate,
                        'fish_weight': fish_weight,
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'month': month,
                        'year': year,
                        'population_imputed_sklearn': population,
                        'ammonia_imputed_sklearn': ammonia,
                        'fish_length_imputed_sklearn': fish_length
                    }
                    
                    # Make prediction
                    if model is not None:
                        input_df = pd.DataFrame([sensor_data])
                        
                        # Ensure all features are present
                        for col in feature_columns:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        
                        input_df = input_df[feature_columns]
                        
                        prediction_encoded = model.predict(input_df)[0]
                        prediction_proba = model.predict_proba(input_df)[0]
                        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                        confidence = prediction_proba[prediction_encoded]
                        
                        # Store in history
                        prediction_record = {
                            'timestamp': datetime.now(),
                            'prediction': prediction,
                            'confidence': confidence,
                            **sensor_data
                        }
                        st.session_state.predictions_history.append(prediction_record)
                        
                        # Display result
                        st.markdown("### Prediction Result")
                        
                        # Determine color based on prediction
                        if prediction == 'Safe':
                            st.markdown('<div class="safe">', unsafe_allow_html=True)
                            st.success(f"✅ Water Quality: **{prediction}**")
                        elif prediction == 'Warning':
                            st.markdown('<div class="warning">', unsafe_allow_html=True)
                            st.warning(f"⚠️ Water Quality: **{prediction}**")
                        else:
                            st.markdown('<div class="danger">', unsafe_allow_html=True)
                            st.error(f"🚨 Water Quality: **{prediction}**")
                        
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show probability distribution
                        prob_df = pd.DataFrame({
                            'Category': label_encoder.classes_,
                            'Probability': prediction_proba
                        })
                        
                        fig = px.bar(prob_df, x='Category', y='Probability', 
                                    color='Probability', color_continuous_scale='RdYlGn',
                                    title='Prediction Probability Distribution')
                        st.plotly_chart(fig, use_container_width=True)
        
        elif input_method == "Simulate Arduino Data":
            st.markdown("### Simulating Arduino Sensor Data")
            
            if st.button("Generate Random Sensor Data"):
                # Generate random sensor data within reasonable ranges
                sensor_data = {
                    'temperature': np.random.uniform(20, 30),
                    'turbidity': np.random.uniform(1, 5),
                    'dissolved_oxygen': np.random.uniform(6, 9),
                    'ph': np.random.uniform(6.5, 8.5),
                    'nitrate': np.random.uniform(5, 20),
                    'fish_weight': np.random.uniform(300, 700),
                    'hour': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),
                    'month': datetime.now().month,
                    'year': 2024,
                    'population_imputed_sklearn': np.random.uniform(800, 1200),
                    'ammonia_imputed_sklearn': np.random.uniform(0.01, 0.1),
                    'fish_length_imputed_sklearn': np.random.uniform(20, 30)
                }
                
                # Display the generated data
                st.write("Generated Sensor Data:")
                st.json(sensor_data)
                
                # Auto-predict with this data
                if model is not None:
                    input_df = pd.DataFrame([sensor_data])
                    
                    for col in feature_columns:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    
                    input_df = input_df[feature_columns]
                    
                    prediction_encoded = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0]
                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                    confidence = prediction_proba[prediction_encoded]
                    
                    # Store prediction
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'prediction': prediction,
                        'confidence': confidence,
                        **sensor_data
                    }
                    st.session_state.predictions_history.append(prediction_record)
                    
                    # Display result
                    st.metric("Prediction", prediction, delta=f"{confidence:.1%} confidence")
        
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file with sensor data", type=['csv'])
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())
                
                if st.button("Predict All Rows"):
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        sensor_data = row.to_dict()
                        
                        if model is not None:
                            input_df = pd.DataFrame([sensor_data])
                            
                            for col in feature_columns:
                                if col not in input_df.columns:
                                    input_df[col] = 0
                            
                            input_df = input_df[feature_columns]
                            
                            prediction_encoded = model.predict(input_df)[0]
                            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                            
                            predictions.append(prediction)
                    
                    df['prediction'] = predictions
                    st.write("Predictions Added:")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="water_quality_predictions.csv",
                        mime="text/csv"
                    )
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Current Status</h2>', unsafe_allow_html=True)
        
        if st.session_state.predictions_history:
            latest = st.session_state.predictions_history[-1]
            
            # Display metrics
            st.markdown("### Latest Reading")
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Temperature", f"{latest['temperature']:.1f}°C")
            with col2_2:
                st.metric("pH", f"{latest['ph']:.1f}")
            with col2_3:
                st.metric("DO", f"{latest['dissolved_oxygen']:.1f} mg/L")
            
            # Gauge chart for water quality
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = latest['confidence'] * 100,
                title = {'text': f"Confidence: {latest['prediction']}"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': latest['confidence'] * 100
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter status
            st.markdown("### Parameter Status")
            
            params = [
                ("pH", latest['ph'], 6.5, 8.5),
                ("Temperature", latest['temperature'], 20, 30),
                ("Dissolved Oxygen", latest['dissolved_oxygen'], 6, 9),
                ("Turbidity", latest['turbidity'], 0, 5)
            ]
            
            for param, value, min_val, max_val in params:
                if min_val <= value <= max_val:
                    st.success(f"✅ {param}: {value:.1f}")
                else:
                    st.error(f"❌ {param}: {value:.1f} (Optimal: {min_val}-{max_val})")
        else:
            st.info("No predictions yet. Submit sensor data to see results here.")

# Page 2: Historical Data
elif page == "Historical Data":
    st.markdown('<h2 class="sub-header">📋 Prediction History</h2>', unsafe_allow_html=True)
    
    if st.session_state.predictions_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Display data
        st.dataframe(history_df.sort_values('timestamp', ascending=False).head(20))
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            safe_count = len(history_df[history_df['prediction'] == 'Safe'])
            st.metric("Safe Predictions", safe_count)
        
        with col2:
            warning_count = len(history_df[history_df['prediction'] == 'Warning'])
            st.metric("Warning Predictions", warning_count)
        
        with col3:
            danger_count = len(history_df[history_df['prediction'] == 'Danger'])
            st.metric("Danger Predictions", danger_count)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series of predictions
            if 'timestamp' in history_df.columns:
                history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                daily_counts = history_df.groupby(['date', 'prediction']).size().reset_index(name='count')
                
                fig = px.line(daily_counts, x='date', y='count', color='prediction',
                            title='Predictions Over Time')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart of prediction distribution
            pred_counts = history_df['prediction'].value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                        title='Prediction Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Export data
        if st.button("Export History to CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="water_quality_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No historical data available yet. Make some predictions first.")

# Page 3: Model Performance
elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">🔬 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Details")
        st.markdown("""
        **Algorithm:** Random Forest Classifier  
        **Number of Trees:** 100  
        **Max Depth:** 10  
        **Training Samples:** 724,642  
        **Testing Samples:** 181,161  
        **Features Used:** 13  
        **Model Size:** 45 MB
        """)
        
        # Feature importance visualization
        if model is not None:
            st.markdown("### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_importance.tail(10), 
                        x='Importance', y='Feature', 
                        orientation='h',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Metrics")
        
        # Mock performance metrics (replace with your actual metrics)
        metrics = {
            "Accuracy": 0.952,
            "Precision": 0.948,
            "Recall": 0.951,
            "F1-Score": 0.949
        }
        
        for metric, value in metrics.items():
            st.progress(value, text=f"{metric}: {value:.1%}")
        
        st.markdown("### Confusion Matrix")
        # Mock confusion matrix (replace with your actual)
        confusion_matrix = np.array([[1200, 50, 10],
                                     [45, 1100, 30],
                                     [15, 25, 1150]])
        
        fig = px.imshow(confusion_matrix,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Safe', 'Warning', 'Danger'],
                       y=['Safe', 'Warning', 'Danger'],
                       text_auto=True,
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# Page 4: About
elif page == "About":
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    This dashboard is part of a water quality monitoring system that uses Arduino sensors and 
    machine learning to predict water safety in real-time.
    
    ### How It Works
    1. **Data Collection**: Arduino sensors measure various water parameters
    2. **Prediction**: Random Forest model analyzes the sensor data
    3. **Visualization**: Results are displayed in this dashboard
    
    ### Parameters Measured
    - **Temperature**: Water temperature in °C
    - **Turbidity**: Water clarity measurement
    - **Dissolved Oxygen**: Oxygen available for aquatic life
    - **pH**: Acidity/alkalinity level
    - **Nitrate**: Nutrient level
    - **Ammonia**: Toxic compound level
    
    ### Safety Categories
    - **Safe**: All parameters within optimal ranges
    - **Warning**: Some parameters require attention
    - **Danger**: Immediate action needed
    
    ### Technology Stack
    - **Backend**: Python, Scikit-learn, Random Forest
    - **Frontend**: Streamlit, Plotly
    - **Hardware**: Arduino with various water quality sensors
    - **Deployment**: Streamlit Cloud
    
    ### Contact & Support
    For questions or support, please contact the project team.
    
    ### Version Information
    - Dashboard Version: 1.0.0
    - Model Version: 1.0.0
    - Last Updated: October 2024
    """)
    
    st.markdown("---")
    st.markdown("**Disclaimer**: This is a prototype system for educational purposes. Always verify critical water quality measurements with certified laboratory tests.")

# Auto-refresh functionality
if auto_refresh and page == "Real-time Prediction":
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray;">
    Water Quality Monitoring System • Powered by Random Forest ML • 
    Data updates every 10 seconds when auto-refresh is enabled
    </div>
    """,
    unsafe_allow_html=True
)