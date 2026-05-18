import streamlit as st

st.set_page_config(
    page_title="Machine Learning Pipeline",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Automated Machine Learning Pipeline")
st.markdown("""
Welcome to the Automated Machine Learning Pipeline! 
Please navigate using the sidebar to go through the different steps of your ML project:
1. **Upload Data**: Upload your raw CSV or Excel files.
2. **EDA & Visualization**: Explore your data visually.
3. **Preprocessing**: Clean, transform, scale, and prepare your data.
4. **Model Training**: Train various ML models.
5. **Evaluation**: Compare model performances.
""")

# Initialize Session State Variables if they don't exist
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_preprocessed' not in st.session_state:
    st.session_state['df_preprocessed'] = None
if 'trained_models_summary' not in st.session_state:
    st.session_state['trained_models_summary'] = []
