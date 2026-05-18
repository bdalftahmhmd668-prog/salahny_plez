import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

# ====================================================================
# 🔥 SECTION 1: HISTORICAL LEADERBOARD (الشاشة الرئيسية القديمة فوق)
# ====================================================================
st.title("📊 Model Performance Evaluation Leaderboard")

if 'trained_models_summary' not in st.session_state or not st.session_state['trained_models_summary']:
    st.warning("⚠️ No historical models trained yet. Models trained from Page 4 will appear here.")
else:
    full_df = pd.DataFrame(st.session_state['trained_models_summary'])
    
    # فلتر نوع المهمة للشاشة الرئيسية
    task_types = full_df["Task Type"].unique()
    selected_task = st.selectbox("Select History Task Type to View:", task_types, key="history_task_filter")
    
    filtered_df = full_df[full_df["Task Type"] == selected_task].reset_index(drop=True)
    st.subheader(f"🏆 {selected_task} Historical Comparison Table")
    
    # عرض العواميد ديناميكياً بناءً على اختيار المستخدم لمنع ظهور None
    if selected_task == "Regression":
        cols_to_show = ["Model Name", "R2 Score", "MAE", "MSE"]
    elif selected_task == "Classification":
        cols_to_show = ["Model Name", "Accuracy", "Precision", "Recall", "F1-Score"]
    else:
        cols_to_show = ["Model Name", "Inertia"]
        
    cols_to_show = [c for c in cols_to_show if c in filtered_df.columns]
    st.dataframe(filtered_df[cols_to_show], use_container_width=True)
    
    # زر مسح التاريخ القديم
    if st.button("Clear Trained Models History"):
        st.session_state['trained_models_summary'] = []
        st.rerun()

# ====================================================================
# 🚀 SECTION 2: LIVE PERFORMANCE SIMULATOR (الميزة الجديدة مضافة في الأسفل)
# ====================================================================
st.write("---") # فاصل أفقي شيك للفصل بين الميزتين
st.header("🎯 Live 5-Model Performance Simulator")

if 'df_preprocessed' not in st.session_state or st.session_state['df_preprocessed'] is None:
    st.warning("⚠️ No preprocessed dataset available. Please complete Page 3 preprocessing first.")
else:
    # حماية الداتا الأصلية بأخذ نسخة معزولة تماماً في الذاكرة
    df_clean = st.session_state['df_preprocessed'].copy()
    
    # صندوق اختيار التارجت الحي مضاف أسفل الشاشة الرئيسية
    target_col = st.selectbox("Select Target Column to evaluate live (On-the-fly):", df_clean.columns, key="live_target_selector")
    
    if target_col:
        # الكشف التلقائي عن نوع المشكلة للحساب الفوري
        unique_count = df_clean[target_col].nunique()
        if unique_count <= 10:
            live_task = "Classification"
            live_metric = "Accuracy"
        else:
            live_task = "Regression"
            live_metric = "R2 Score"
            
        st.caption(f"🤖 **Live Mode Auto-Detected:** {live_task} (Scoring via {live_metric})")
        
        # تجهيز المصفوفات من النسخة الآمنة المعزولة
        X_live = df_clean.drop(columns=[target_col])
        y_live = df_clean[target_col]
        X_live = pd.get_dummies(X_live, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X_live, y_live, test_size=0.2, random_state=42)
        
        # تجهيز حوض الـ 5 موديلات المطلوبة بالترتيب والمسميات المختصرة
        if live_task == "Classification":
            models_pool = {
                "XGB": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                "DT": DecisionTreeClassifier(random_state=42),
                "RF": RandomForestClassifier(random_state=42),
                "LR": LogisticRegression(max_iter=1000, random_state=42),
                "KNN": KNeighborsClassifier()
            }
        else:
            models_pool = {
                "XGB": xgb.XGBRegressor(random_state=42),
                "DT": DecisionTreeRegressor(random_state=42),
                "RF": RandomForestRegressor(random_state=42),
                "LR": LinearRegression(),
                "KNN": KNeighborsRegressor()
            }
            
        console_output = ""
        
        # حساب فوري في أجزاء من الثانية خلف الكواليس
        with st.spinner("Simulating performance across all 5 models..."):
            for short_name, model_obj in models_pool.items():
                try:
                    model_obj.fit(X_train, y_train)
                    preds = model_obj.predict(X_test)
                    
                    score_val = accuracy_score(y_test, preds) if live_task == "Classification" else r2_score(y_test, preds)
                    console_output += f"{short_name} {score_val:.16f}\n"
                except:
                    console_output += f"{short_name} Error: Skipped for this target.\n"
                    
        # عرض الكونسول النصية النظيفة مطابقة للـ سكرين تحت الجدول الرئيسي
        st.subheader("📋 Raw Performance Scores Console")
        st.code(console_output, language="text")
