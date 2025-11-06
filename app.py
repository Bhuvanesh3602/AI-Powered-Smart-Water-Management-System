import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from data_manager import DataManager
from anomaly_detector import AnomalyDetector
from predictor import WaterUsagePredictor
from gemini_helper import (
    generate_water_conservation_tips,
    answer_water_usage_question,
    predict_water_usage_insights,
    analyze_anomaly
)

st.set_page_config(
    page_title="AquaMind - Smart Water Management",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def initialize_components():
    data_manager = DataManager()
    anomaly_detector = AnomalyDetector(contamination=0.05)
    predictor_rf = WaterUsagePredictor(model_type='random_forest')
    predictor_lstm = WaterUsagePredictor(model_type='lstm')
    
    anomaly_detector.load_model()
    predictor_rf.load_model()
    predictor_lstm.load_model()
    
    return data_manager, anomaly_detector, predictor_rf, predictor_lstm

@st.cache_data
def load_and_process_data(_data_manager):
    df = _data_manager.load_dataset()
    if df is None:
        return None, None
    
    flow_col = _data_manager._get_flow_column()
    return df, flow_col

def prepare_df_for_display(df):
    df_display = df.copy()
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = df_display['timestamp'].astype(str)
    for col in df_display.select_dtypes(include=['datetime64']).columns:
        df_display[col] = df_display[col].astype(str)
    return df_display

def main():
    st.title("💧 AquaMind - AI-Powered Smart Water Management System")
    st.markdown("Monitor, analyze, and predict water usage with machine learning and intelligent analytics")
    
    data_manager, anomaly_detector, predictor_rf, predictor_lstm = initialize_components()
    
    with st.spinner("Loading water usage data from Kaggle..."):
        df, flow_col = load_and_process_data(data_manager)
    
    if df is None:
        st.error("Failed to load dataset. Please check your internet connection and try again.")
        return
    
    if flow_col is None:
        st.error("Could not identify flow rate column in the dataset.")
        return
    
    st.success(f"✅ Loaded {len(df):,} water usage records")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🚨 Anomaly Detection", 
        "📈 Predictions", 
        "🤖 AI Assistant", 
        "🏷️ Data Labeling",
        "📋 Raw Data"
    ])
    
    with tab1:
        show_dashboard(df, flow_col, data_manager)
    
    with tab2:
        show_anomaly_detection(df, flow_col, anomaly_detector, data_manager)
    
    with tab3:
        show_predictions(df, flow_col, predictor_rf, predictor_lstm)
    
    with tab4:
        show_ai_assistant(df, flow_col, data_manager, anomaly_detector)
    
    with tab5:
        show_data_labeling(df, data_manager)
    
    with tab6:
        show_raw_data(df, data_manager)

def show_dashboard(df, flow_col, data_manager):
    st.header("📊 Water Usage Dashboard")
    
    stats = data_manager.get_usage_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{stats.get('total_records', 0):,}")
    with col2:
        st.metric("Avg Flow Rate", f"{stats.get('avg_flow_rate', 0):.2f} L/min")
    with col3:
        st.metric("Max Flow Rate", f"{stats.get('max_flow_rate', 0):.2f} L/min")
    with col4:
        avg_daily = stats.get('avg_daily_usage', 0)
        st.metric("Avg Daily Usage", f"{avg_daily:.0f} L")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Water Usage Trend")
        daily_usage = data_manager.get_daily_usage()
        
        if not daily_usage.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_usage['date'],
                y=daily_usage['total_usage'],
                mode='lines+markers',
                name='Total Daily Usage',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Water Usage (Liters)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily usage data available")
    
    with col2:
        st.subheader("Hourly Usage Pattern")
        hourly_pattern = data_manager.get_hourly_pattern()
        
        if not hourly_pattern.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_pattern['hour'],
                y=hourly_pattern['avg_flow_rate'],
                marker_color='#2ca02c',
                name='Avg Flow Rate'
            ))
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Avg Flow Rate (L/min)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly pattern data available")
    
    st.subheader("Flow Rate Distribution")
    fig = px.histogram(
        df, 
        x=flow_col, 
        nbins=50,
        title="Distribution of Flow Rates",
        labels={flow_col: "Flow Rate (L/min)", "count": "Frequency"}
    )
    fig.update_traces(marker_color='#ff7f0e')
    st.plotly_chart(fig, use_container_width=True)
    
    if 'timestamp' in df.columns:
        st.subheader("Time Series Flow Rate")
        
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sample['timestamp'],
            y=df_sample[flow_col],
            mode='lines',
            name='Flow Rate',
            line=dict(color='#9467bd', width=1)
        ))
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Flow Rate (L/min)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df, flow_col, anomaly_detector, data_manager):
    st.header("🚨 Anomaly & Leak Detection")
    
    st.markdown("Using **Isolation Forest** algorithm to detect unusual water usage patterns and potential leaks.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        contamination = st.slider(
            "Contamination Rate (Expected % of anomalies)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Percentage of data expected to be anomalies"
        )
        
        threshold_multiplier = st.slider(
            "Leak Threshold Multiplier",
            min_value=1.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="How many standard deviations above normal to consider a leak"
        )
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Training anomaly detection model..."):
                anomaly_detector.contamination = contamination
                anomaly_detector.train(df, flow_col)
                st.session_state['anomaly_trained'] = True
                st.success("✅ Model trained successfully!")
    
    with col2:
        if 'anomaly_trained' not in st.session_state:
            st.info("👈 Click 'Detect Anomalies' to start the analysis")
        else:
            with st.spinner("Detecting anomalies..."):
                df_anomalies = anomaly_detector.detect_anomalies(df, flow_col)
                summary = anomaly_detector.get_anomaly_summary(df_anomalies, flow_col)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Anomalies", summary['total_anomalies'])
                with col_b:
                    st.metric("Anomaly Rate", f"{summary['anomaly_percentage']:.2f}%")
                with col_c:
                    if summary['total_anomalies'] > 0:
                        st.metric("Avg Anomaly Flow", f"{summary.get('avg_anomaly_flow', 0):.2f} L/min")
                
                leak_alerts = anomaly_detector.get_leak_alerts(
                    df_anomalies, 
                    flow_col, 
                    threshold_multiplier
                )
                
                if leak_alerts:
                    st.subheader("🚨 Leak Alerts")
                    for alert in leak_alerts[:5]:
                        severity_color = "🔴" if alert['severity'] == 'Critical' else "🟡"
                        st.warning(
                            f"{severity_color} **{alert['message']}**\n\n"
                            f"Time: {alert['timestamp']} | "
                            f"Flow Rate: {alert['flow_rate']:.2f} L/min | "
                            f"Normal: {alert['normal_flow']:.2f} L/min"
                        )
                    
                    if len(leak_alerts) > 5:
                        st.info(f"+ {len(leak_alerts) - 5} more leak alerts")
                else:
                    st.success("✅ No critical leaks detected!")
                
                st.subheader("Anomaly Visualization")
                
                sample_size = min(1000, len(df_anomalies))
                df_viz = df_anomalies.sample(n=sample_size, random_state=42)
                
                if 'timestamp' in df_viz.columns:
                    df_viz = df_viz.sort_values('timestamp')
                
                fig = go.Figure()
                
                normal_data = df_viz[df_viz['is_anomaly'] == False]
                anomaly_data = df_viz[df_viz['is_anomaly'] == True]
                
                if len(normal_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=normal_data.index if 'timestamp' not in df_viz.columns else normal_data['timestamp'],
                        y=normal_data[flow_col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='green', size=4, opacity=0.6)
                    ))
                
                if len(anomaly_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data.index if 'timestamp' not in df_viz.columns else anomaly_data['timestamp'],
                        y=anomaly_data[flow_col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                
                fig.update_layout(
                    xaxis_title="Time" if 'timestamp' in df_viz.columns else "Index",
                    yaxis_title="Flow Rate (L/min)",
                    hovermode='closest',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if summary['total_anomalies'] > 0 and len(leak_alerts) > 0:
                    st.subheader("🤖 AI Anomaly Analysis")
                    
                    if st.button("Get AI Analysis of Top Anomaly"):
                        with st.spinner("Analyzing anomaly with AI..."):
                            top_alert = leak_alerts[0]
                            
                            normal_data = df_anomalies[df_anomalies['is_anomaly'] == False]
                            normal_mean = normal_data[flow_col].mean()
                            normal_std = normal_data[flow_col].std()
                            
                            anomaly_data = {
                                'flow_rate': top_alert['flow_rate'],
                                'normal_range': f"{normal_mean:.2f} ± {normal_std:.2f}",
                                'timestamp': str(top_alert['timestamp']),
                                'severity': top_alert['severity']
                            }
                            
                            analysis = analyze_anomaly(anomaly_data)
                            st.info(analysis)

def show_predictions(df, flow_col, predictor_rf, predictor_lstm):
    st.header("📈 Water Usage Predictions")
    
    st.markdown("Forecast future water consumption using **Random Forest** and **LSTM** models.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_type = st.radio(
            "Select Prediction Model",
            options=['Random Forest', 'LSTM'],
            help="Random Forest is faster, LSTM may be more accurate for time series"
        )
        
        days_to_predict = st.slider(
            "Days to Predict",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of future days to forecast"
        )
        
        predictor = predictor_rf if model_type == 'Random Forest' else predictor_lstm
        
        if st.button("🔮 Train & Predict", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                if model_type == 'Random Forest':
                    metrics = predictor.train_random_forest(df, flow_col)
                else:
                    metrics = predictor.train_lstm(df, flow_col, epochs=30)
                
                st.session_state['prediction_trained'] = True
                st.session_state['prediction_metrics'] = metrics
                st.session_state['predictor_type'] = model_type
                st.success("✅ Model trained successfully!")
    
    with col2:
        if 'prediction_trained' not in st.session_state:
            st.info("👈 Select a model and click 'Train & Predict' to start forecasting")
        else:
            metrics = st.session_state.get('prediction_metrics', {})
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Train RMSE", f"{metrics.get('train_rmse', 0):.2f}")
            with col_b:
                st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}")
            with col_c:
                st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f}")
            
            with st.spinner("Generating predictions..."):
                pred_df = predictor.get_prediction_dataframe(df, flow_col, days_to_predict)
                
                if pred_df is not None:
                    st.subheader("Future Water Usage Forecast")
                    
                    daily_usage = data_manager.get_daily_usage()
                    
                    fig = go.Figure()
                    
                    if not daily_usage.empty:
                        recent_days = 30
                        daily_recent = daily_usage.tail(recent_days)
                        
                        fig.add_trace(go.Scatter(
                            x=daily_recent['date'],
                            y=daily_recent['total_usage'],
                            mode='lines+markers',
                            name='Historical Usage',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted_usage'],
                        mode='lines+markers',
                        name='Predicted Usage',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Water Usage (Liters)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Predicted Values")
                    pred_df_display = prepare_df_for_display(pred_df)
                    st.dataframe(pred_df_display, use_container_width=True)
                    
                    st.subheader("🤖 AI Prediction Insights")
                    
                    if st.button("Get AI Insights on Predictions"):
                        with st.spinner("Generating insights..."):
                            if not daily_usage.empty:
                                recent_usage = daily_usage.tail(7)['total_usage'].tolist()
                            else:
                                recent_usage = [0] * 7
                            
                            avg_prediction = pred_df['predicted_usage'].mean()
                            
                            insights = predict_water_usage_insights(recent_usage, avg_prediction)
                            st.info(insights)
                else:
                    st.error("Failed to generate predictions. Please ensure there is enough data.")

def show_ai_assistant(df, flow_col, data_manager, anomaly_detector):
    st.header("🤖 AI Assistant - AquaMind Chat")
    
    st.markdown("Ask questions about your water usage or get personalized conservation tips.")
    
    tab_chat, tab_tips = st.tabs(["💬 Chat Assistant", "💡 Conservation Tips"])
    
    with tab_chat:
        st.subheader("Ask AquaMind")
        
        stats = data_manager.get_usage_statistics()
        
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., How much water did I use last week? When was my last anomaly?"
        )
        
        if st.button("Ask", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    context_data = {
                        'total_records': stats.get('total_records', 0),
                        'date_range': stats.get('date_range', 'N/A'),
                        'avg_usage': stats.get('avg_daily_usage', 0),
                        'anomalies_count': 0,
                        'last_anomaly_date': 'None detected'
                    }
                    
                    if 'anomaly_trained' in st.session_state:
                        df_anomalies = anomaly_detector.detect_anomalies(df, flow_col)
                        summary = anomaly_detector.get_anomaly_summary(df_anomalies, flow_col)
                        context_data['anomalies_count'] = summary['total_anomalies']
                        if summary.get('last_anomaly'):
                            context_data['last_anomaly_date'] = str(summary['last_anomaly'])
                    
                    answer = answer_water_usage_question(question, context_data)
                    st.success("**AquaMind:** " + answer)
            else:
                st.warning("Please enter a question")
        
        st.markdown("---")
        st.markdown("**Example questions:**")
        st.markdown("- What's my average daily water usage?")
        st.markdown("- How many anomalies were detected?")
        st.markdown("- What time of day do I use the most water?")
        st.markdown("- Are there any patterns in my water consumption?")
    
    with tab_tips:
        st.subheader("Personalized Water Conservation Tips")
        
        if st.button("Generate Conservation Tips 💡", type="primary"):
            with st.spinner("Generating personalized tips..."):
                stats = data_manager.get_usage_statistics()
                hourly = data_manager.get_hourly_pattern()
                
                usage_data = {
                    'avg_daily_usage': stats.get('avg_daily_usage', 0),
                    'peak_time': f"{hourly['hour'].iloc[hourly['avg_flow_rate'].idxmax()]}:00" if not hourly.empty else 'N/A',
                    'anomalies_count': 0,
                    'trend': 'stable'
                }
                
                if 'anomaly_trained' in st.session_state:
                    df_anomalies = anomaly_detector.detect_anomalies(df, flow_col)
                    summary = anomaly_detector.get_anomaly_summary(df_anomalies, flow_col)
                    usage_data['anomalies_count'] = summary['total_anomalies']
                
                tips = generate_water_conservation_tips(usage_data)
                st.success(tips)

def show_data_labeling(df, data_manager):
    st.header("🏷️ Data Labeling")
    
    st.markdown("Add custom labels to water consumption records for better tracking and analysis.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add Label to Record")
        
        record_index = st.number_input(
            "Record Index",
            min_value=0,
            max_value=len(df)-1,
            value=0,
            help="Index of the record to label"
        )
        
        if record_index < len(df):
            st.write("**Record Details:**")
            record = df.iloc[record_index]
            st.json(record.to_dict())
        
        label_text = st.text_input(
            "Label",
            placeholder="e.g., Leak fixed, New appliance installed, Vacation period"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Add Label", type="primary"):
                if label_text:
                    data_manager.add_label(int(record_index), label_text)
                    st.success(f"✅ Label added to record {record_index}")
                else:
                    st.warning("Please enter a label")
        
        with col_btn2:
            if st.button("Remove Label"):
                data_manager.remove_label(int(record_index))
                st.success(f"✅ Label removed from record {record_index}")
    
    with col2:
        st.subheader("Current Labels")
        
        if data_manager.user_labels:
            st.write(f"**Total Labels:** {len(data_manager.user_labels)}")
            
            for idx, label_data in list(data_manager.user_labels.items())[:10]:
                st.text(f"[{idx}] {label_data['label']}")
            
            if len(data_manager.user_labels) > 10:
                st.info(f"+ {len(data_manager.user_labels) - 10} more labels")
        else:
            st.info("No labels added yet")
    
    st.markdown("---")
    
    if st.button("Export Labeled Data"):
        df_export = data_manager.export_labeled_data()
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"labeled_water_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.success("✅ Data ready for download!")

def show_raw_data(df, data_manager):
    st.header("📋 Raw Data Explorer")
    
    st.markdown(f"**Total Records:** {len(df):,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_rows = st.slider("Number of rows to display", 10, 1000, 100)
    
    with col2:
        view_mode = st.radio("View Mode", ["Head", "Tail", "Sample"])
    
    if view_mode == "Head":
        df_view = prepare_df_for_display(df.head(num_rows))
        st.dataframe(df_view, use_container_width=True)
    elif view_mode == "Tail":
        df_view = prepare_df_for_display(df.tail(num_rows))
        st.dataframe(df_view, use_container_width=True)
    else:
        df_view = prepare_df_for_display(df.sample(min(num_rows, len(df))))
        st.dataframe(df_view, use_container_width=True)
    
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Names:**")
        st.write(list(df.columns))
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.to_dict())
    
    st.subheader("Statistical Summary")
    df_stats = df.select_dtypes(include=[np.number]).describe()
    st.dataframe(df_stats, use_container_width=True)

if __name__ == "__main__":
    data_manager, _, _, _ = initialize_components()
    main()
