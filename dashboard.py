import streamlit as st
import pandas as pd
from evaluate import analyze_tripinfo # We reuse the function from your evaluate.py script

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Traffic Management Dashboard",
    page_icon="🚦",
    layout="wide"
)

# --- Main Title ---
st.title("🚦 AI Smart Traffic Management System")
st.markdown("An analysis of a Multi-Agent Reinforcement Learning model for urban traffic optimization.")

# --- Load Data ---
# We call the analysis function for both result files
ai_duration, ai_wait_time = analyze_tripinfo("tripinfo_trained_ai.xml")
fixed_duration, fixed_wait_time = analyze_tripinfo("tripinfo_fixed.xml")

# --- Display Results ---
st.header("Performance Comparison 📊")

if ai_duration is not None and fixed_duration is not None:
    # Calculate improvements
    duration_improvement = ((fixed_duration - ai_duration) / fixed_duration) * 100
    wait_time_improvement = ((fixed_wait_time - ai_wait_time) / fixed_wait_time) * 100

    # Use columns for a side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Commute Time")
        st.metric(
            label="Fixed-Timer (Baseline)",
            value=f"{fixed_duration:.2f} s"
        )
        st.metric(
            label="AI Model (Trained)",
            value=f"{ai_duration:.2f} s",
            delta=f"-{duration_improvement:.2f}% improvement",
            delta_color="normal" # "normal" shows green for negative delta
        )

    with col2:
        st.subheader("Average Waiting Time")
        st.metric(
            label="Fixed-Timer (Baseline)",
            value=f"{fixed_wait_time:.2f} s"
        )
        st.metric(
            label="AI Model (Trained)",
            value=f"{ai_wait_time:.2f} s",
            delta=f"-{wait_time_improvement:.2f}% improvement",
            delta_color="normal"
        )

    st.success(f"**Key Finding:** The AI model dramatically reduced vehicle waiting time by over {wait_time_improvement:.0f}%, leading to a smoother overall traffic flow.")

    # --- Create a Bar Chart for Visualization ---
    st.header("Visual Comparison")
    chart_data = pd.DataFrame({
        'Metric': ['Avg. Commute Time (s)', 'Avg. Wait Time (s)'],
        'Fixed-Timer Baseline': [fixed_duration, fixed_wait_time],
        'AI Model': [ai_duration, ai_wait_time]
    })
    st.bar_chart(chart_data.set_index('Metric'), height=400)

else:
    st.error("Could not find result files. Please run the evaluation scripts first.")