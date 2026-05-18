import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Unsupervised Models
from sklearn.cluster import KMeans

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")
st.title("🤖 Model Training")

if 'df_preprocessed' in st.session_state and st.session_state['df_preprocessed'] is not None:
    df = st.session_state['df_preprocessed']
    
    # Initialize the summary list in session state if it doesn't exist
    if 'trained_models_summary' not in st.session_state:
        st.session_state['trained_models_summary'] = []
        
    task_type = st.radio("Select Task Type", ["Classification", "Regression", "Unsupervised Learning"])
    
    # Model selection based on task type
    if task_type == "Classification":
        model_name = st.selectbox("Select Classification Model:", ["Random Forest (Recommended)", "Logistic Regression", "Decision Tree", "SVM", "KNN", "Naive Bayes", "Neural Network (MLP)"])
    elif task_type == "Regression":
        model_name = st.selectbox("Select Regression Model:", ["Random Forest Regressor (Recommended)", "Linear Regression", "XGBoost Regressor"])
    else:
        model_name = st.selectbox("Select Model", ["KMeans"])

    target_col = None
    if task_type != "Unsupervised Learning":
        target_col = st.selectbox("Select Target Column", df.columns)
        
    if st.button("Train Model"):
        with st.spinner("Training the model, please wait..."):
            try:
                if task_type == "Unsupervised Learning":
                    X = df
                    
                    # Ensure numerical data only for KMeans
                    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
                        st.warning("KMeans requires all numerical features. Categorical columns found. Please encode them in Preprocessing.")
                        st.stop()
                        
                    model = KMeans(n_clusters=3, random_state=42)
                    predictions = model.fit_predict(X)
                    
                    # Save results
                    result = {
                        "Model Name": model_name,
                        "Task Type": task_type,
                        "Inertia": round(model.inertia_, 4),
                        "Metric Name": "Inertia",
                        "Score": round(model.inertia_, 4)
                    }
                    
                    st.success(f"{model_name} trained successfully!")
                    st.write(f"**Inertia**: {result['Inertia']}")
                    
                    # Remove older records of the same model to prevent duplicates
                    st.session_state['trained_models_summary'] = [
                        m for m in st.session_state['trained_models_summary'] if m["Model Name"] != model_name
                    ]
                    st.session_state['trained_models_summary'].append(result)
                    
                else:
                    if target_col is None:
                        st.error("Please select a target column.")
                        st.stop()
                        
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Instantiate Model
                    if model_name.startswith("Logistic Regression"):
                        model = LogisticRegression(max_iter=1000)
                    elif model_name.startswith("Decision Tree"):
                        model = DecisionTreeClassifier(random_state=42)
                    elif model_name.startswith("Random Forest"):
                        if task_type == "Classification":
                            model = RandomForestClassifier(random_state=42)
                        else:
                            model = RandomForestRegressor(random_state=42)
                    elif model_name.startswith("SVM"):
                        model = SVC()
                    elif model_name.startswith("KNN"):
                        model = KNeighborsClassifier()
                    elif model_name.startswith("Naive Bayes"):
                        model = GaussianNB()
                    elif model_name.startswith("Neural Network"):
                        model = MLPClassifier(max_iter=1000, random_state=42)
                    elif model_name.startswith("Linear Regression"):
                        model = LinearRegression()
                    elif model_name.startswith("XGBoost"):
                        from xgboost import XGBRegressor
                        model = XGBRegressor(random_state=42)
                        
                    # Secure execution of model fitting
                    model.fit(X_train, y_train)
                    
                    # If successful, calculate predictions and save metrics
                    y_pred = model.predict(X_test)
                    st.success("🎉 Model trained successfully with no errors!")
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
                    
                    model_results = {"Model Name": model_name, "Task Type": task_type}
                    
                    if task_type == "Classification":
                        model_results["Accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                        model_results["Precision"] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                        model_results["Recall"] = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                        model_results["F1-Score"] = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
                        
                        st.write(f"**Accuracy**: {model_results['Accuracy']}")
                    else:
                        model_results["MAE"] = round(mean_absolute_error(y_test, y_pred), 4)
                        model_results["MSE"] = round(mean_squared_error(y_test, y_pred), 4)
                        model_results["R2 Score"] = round(r2_score(y_test, y_pred), 4)
                        
                        st.write(f"**R2 Score**: {model_results['R2 Score']}")
                        st.write(f"**MSE**: {model_results['MSE']}")
                        
                    # Remove older records of the same model to prevent duplicates
                    st.session_state['trained_models_summary'] = [
                        m for m in st.session_state['trained_models_summary'] if m["Model Name"] != model_name
                    ]
                    st.session_state['trained_models_summary'].append(model_results)
                    st.toast(f"{model_name} added to leaderboard!")

            except ValueError as e:
                error_msg = str(e)
                
                # Catching the exact continuous float target error during classification
                if "Unknown label type: continuous" in error_msg:
                    st.error(
                        f"❌ **Training Blocked:** The selected target column contains continuous decimal values (floats) due to scaling.\n\n"
                        f"💡 **How to Fix This:**\n"
                        f"1. Go back to the **Preprocessing** page and turn off Scaling/Normalization for this target column, OR\n"
                        f"2. Change your **Model Type** to **Regression** on the left sidebar to safely predict these continuous numbers."
                    )
                elif "could not convert string to float" in error_msg:
                    st.error(
                        f"❌ **Data Feature Error:** The model cannot train on raw unencoded text data.\n\n"
                        f"💡 **How to Fix This:** Go back to the **Preprocessing** page and apply **Label Encoding** to your categorical features."
                    )
                else:
                    st.error(f"❌ **Value Error:** {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ **Unexpected Training Error:** {str(e)}")
            
else:
    st.warning("Please complete data preprocessing on the Preprocessing page first.")
