import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", page_icon="📂", layout="wide")

st.title("📂 Upload Data")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Determine the file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.session_state['df'] = df
        st.success('File uploaded and verified successfully!')
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a file to proceed.")
