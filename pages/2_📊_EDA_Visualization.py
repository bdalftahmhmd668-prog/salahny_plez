import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Visualization", page_icon="📊", layout="wide")

st.title("📊 EDA Visualization")

if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df']
    
    st.subheader("Data Visualization")
    
    # Get numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Line Plot", "Box Plot"])
    
    if plot_type in ["Scatter Plot", "Line Plot"]:
        if len(numerical_cols) < 2:
            st.warning("You need at least 2 numerical columns for this plot.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis (Numerical)", numerical_cols, index=0)
            with col2:
                y_axis = st.selectbox("Select Y-axis (Numerical)", numerical_cols, index=1)
                
            if st.button("Generate Plot"):
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                    ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
                else:
                    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                    ax.set_title(f"Line Plot: {x_axis} vs {y_axis}")
                st.pyplot(fig)
                
    elif plot_type == "Box Plot":
        if not categorical_cols:
            st.warning("You need at least 1 categorical column for the X-axis.")
        elif not numerical_cols:
            st.warning("You need at least 1 numerical column for the Y-axis.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis (Categorical)", categorical_cols)
            with col2:
                y_axis = st.selectbox("Select Y-axis (Numerical)", numerical_cols)
                
            if st.button("Generate Plot"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
                ax.set_title(f"Box Plot: {y_axis} grouped by {x_axis}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
else:
    st.warning("Please upload a dataset on the Upload Data page first.")
