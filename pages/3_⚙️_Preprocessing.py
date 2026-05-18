import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats.mstats import winsorize
from scipy import stats

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

st.title("⚙️ Data Preprocessing Pipeline")

if 'df' in st.session_state and st.session_state['df'] is not None:
    df = st.session_state['df'].copy()
    
    st.subheader("Configure Preprocessing Steps")
    
    with st.form("preprocessing_form"):
        # 1. Missing Values
        imputer_choice = st.selectbox("1. Missing Values Imputation:", ["Mean Imputation (Recommended)", "Median Imputation", "Mode Imputation", "KNN Imputer", "Iterative Imputer", "None"])
        
        # 2. Outliers
        outlier_choice = st.selectbox("2. Outlier Handling:", ["Clipping/Capping (Recommended)", "Z-Score Capping", "Winsorization", "None"])
        
        # 3. Encoding
        encoding_choice = st.selectbox("3. Categorical Encoding:", ["Label Encoding (Recommended)", "One-Hot Encoding", "None"])
        
        # 4. Transformation
        transform_choice = st.selectbox("4. Feature Transformation:", ["None (Recommended)", "Log Transformation", "Box-Cox Transformation", "Yeo-Johnson", "Polynomial Features"])
        
        # 5. Scaling
        scaling_choice = st.selectbox("5. Feature Scaling:", ["Standard Scaler (Recommended)", "MinMax Scaler", "None"])
        
        # Target Column Selection for Imbalance & RFE
        st.markdown("### Target Column (Required for Imbalance Handling & RFE)")
        target_col = st.selectbox("Select Target Column (Leave as None if unsupervised)", ["None"] + list(df.columns))
        
        # 6. Imbalance Handling
        imbalance_choice = st.selectbox("6. Imbalance Handling:", ["None (Recommended)", "Oversampling (Pure Pandas)", "Undersampling"])
        
        # 7. Feature Selection / Reduction
        reduction_choice = st.selectbox("7. Feature Selection / Reduction:", ["None (Recommended)", "PCA", "RFE"])
        
        n_features = None
        if not reduction_choice.startswith("None"):
            n_features = st.number_input("Number of components / features to keep", min_value=1, max_value=df.shape[1], value=max(1, df.shape[1]-1))

        submitted = st.form_submit_button("Apply Preprocessing")
        
    if submitted:
        try:
            with st.spinner("Applying preprocessing steps..."):
                processed_df = df.copy()
                
                # Split features and target if target is provided
                if target_col != "None" and target_col in processed_df.columns:
                    X = processed_df.drop(columns=[target_col])
                    y = processed_df[target_col]
                else:
                    X = processed_df
                    y = None
                
                num_cols = X.select_dtypes(include=['number']).columns.tolist()
                cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

                # 1. Missing Values
                if not imputer_choice.startswith("None"):
                    if imputer_choice.startswith("Mean"):
                        if num_cols:
                            num_imputer = SimpleImputer(strategy='mean')
                            X[num_cols] = num_imputer.fit_transform(X[num_cols])
                    elif imputer_choice.startswith("Median"):
                        if num_cols:
                            num_imputer = SimpleImputer(strategy='median')
                            X[num_cols] = num_imputer.fit_transform(X[num_cols])
                    elif imputer_choice.startswith("Mode"):
                        if num_cols:
                            num_imputer = SimpleImputer(strategy='most_frequent')
                            X[num_cols] = num_imputer.fit_transform(X[num_cols])
                    elif imputer_choice.startswith("KNN"):
                        if num_cols:
                            imputer = KNNImputer()
                            X[num_cols] = imputer.fit_transform(X[num_cols])
                    elif imputer_choice.startswith("Iterative"):
                        if num_cols:
                            imputer = IterativeImputer(random_state=42)
                            X[num_cols] = imputer.fit_transform(X[num_cols])
                            
                    # Always mode impute categorical if any imputation is selected
                    if cat_cols:
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

                # --- 1. SAFE OUTLIER HANDLING (Auto-Shielding Discrete & Target Columns) ---
                if not outlier_choice.startswith("None"):
                    # تلقائياً نستثني عمود الـ Target وأي عمود فئات فريدته 10 أو أقل لحمايته من التشوه
                    num_cols_to_process = [
                        c for c in X.select_dtypes(include=[np.number]).columns 
                        if c != target_col and X[c].nunique() > 10
                    ]
                    
                    for col in num_cols_to_process:
                        if outlier_choice.startswith("Clipping"):
                            Q1 = X[col].quantile(0.25)
                            Q3 = X[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                        elif outlier_choice.startswith("Z-Score"):
                            mean_val = X[col].mean()
                            std_val = X[col].std()
                            if std_val > 0:
                                lower_bound = mean_val - 3 * std_val
                                upper_bound = mean_val + 3 * std_val
                                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
                        elif outlier_choice.startswith("Winsorization"):
                            # بديل آمن للـ winsorize باستخدام الـ clip والمقاييس المئوية المسموحة
                            lower_bound = X[col].quantile(0.01)
                            upper_bound = X[col].quantile(0.99)
                            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

                # 3. Encoding
                if not encoding_choice.startswith("None") and cat_cols:
                    if encoding_choice.startswith("Label"):
                        le = LabelEncoder()
                        for col in cat_cols:
                            X[col] = le.fit_transform(X[col].astype(str))
                    elif encoding_choice.startswith("One-Hot"):
                        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                
                # Re-evaluate numerical columns after encoding
                num_cols = X.select_dtypes(include=['number']).columns.tolist()

                # --- 2. SAFE FEATURE TRANSFORMATION (Auto-Shielding Discrete & Target Columns) ---
                if not transform_choice.startswith("None"):
                    num_cols_to_transform = [
                        c for c in X.select_dtypes(include=[np.number]).columns 
                        if c != target_col and X[c].nunique() > 10
                    ]
                    
                    if num_cols_to_transform:
                        if transform_choice.startswith("Log"):
                            for col in num_cols_to_transform:
                                min_val = X[col].min()
                                if min_val < 0:
                                    X[col] = np.log1p(X[col] - min_val)
                                else:
                                    X[col] = np.log1p(X[col])
                        elif transform_choice.startswith("Box-Cox"):
                            pt = PowerTransformer(method='box-cox')
                            for col in num_cols_to_transform:
                                min_val = X[col].min()
                                if min_val <= 0:
                                    X[col] = X[col] - min_val + 1e-5
                            X[num_cols_to_transform] = pt.fit_transform(X[num_cols_to_transform])
                        elif transform_choice.startswith("Yeo-Johnson"):
                            pt = PowerTransformer(method='yeo-johnson')
                            X[num_cols_to_transform] = pt.fit_transform(X[num_cols_to_transform])
                        elif transform_choice.startswith("Polynomial"):
                            poly = PolynomialFeatures(degree=2, include_bias=False)
                            poly_features = poly.fit_transform(X[num_cols_to_transform])
                            poly_feature_names = poly.get_feature_names_out(num_cols_to_transform)
                            X_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
                            X = X.drop(columns=num_cols_to_transform)
                            X = pd.concat([X, X_poly], axis=1)

                # --- 3. SAFE FEATURE SCALING (Auto-Shielding Discrete & Target Columns - CRITICAL FIX) ---
                if scaling_choice.startswith("Standard") or scaling_choice.startswith("MinMax"):
                    # استخراج كافة الأعمدة الرقمية الحقيقية المستمرة (التي تحتوي على أكثر من 10 قيم فريدة)
                    cols_to_scale = [
                        c for c in X.select_dtypes(include=[np.number]).columns 
                        if c != target_col and X[c].nunique() > 10
                    ]
                    
                    if cols_to_scale:
                        if "Standard" in scaling_choice:
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                            
                        # نعمل scaling للأعمدة المستمرة فقط ونترك أعمدة الفئات والسنين صحيحة تماماً
                        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

                # 6. Imbalance Handling
                if not imbalance_choice.startswith("None") and y is not None:
                    # Imbalance handling requires target variable and mostly applies to classification
                    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                        st.warning("Cannot apply imbalance handling because there are still missing values. Please select an Imputation method.")
                    else:
                        if imbalance_choice.startswith("Oversampling"):
                            # --- Safe Native Pandas Resampling (Replaces imblearn completely) ---
                            temp_df = pd.DataFrame(X).copy()
                            temp_df['target_label'] = y
                            
                            max_class_size = temp_df['target_label'].value_counts().max()
                            balanced_chunks = []
                            
                            for class_val in temp_df['target_label'].unique():
                                class_subset = temp_df[temp_df['target_label'] == class_val]
                                resampled_chunk = class_subset.sample(n=max_class_size, replace=True, random_state=42)
                                balanced_chunks.append(resampled_chunk)
                                
                            balanced_df = pd.concat(balanced_chunks, axis=0).reset_index(drop=True)
                            X = balanced_df.drop(columns=['target_label'])
                            y = balanced_df['target_label']
                            st.sidebar.success("🎉 Balanced via Pure Pandas Oversampling!")
                            
                        elif imbalance_choice.startswith("Undersampling"):
                            # --- Safe Native Pandas Resampling (Replaces imblearn completely) ---
                            temp_df = pd.DataFrame(X).copy()
                            temp_df['target_label'] = y
                            
                            min_class_size = temp_df['target_label'].value_counts().min()
                            balanced_chunks = []
                            
                            for class_val in temp_df['target_label'].unique():
                                class_subset = temp_df[temp_df['target_label'] == class_val]
                                resampled_chunk = class_subset.sample(n=min_class_size, replace=False, random_state=42)
                                balanced_chunks.append(resampled_chunk)
                                
                            balanced_df = pd.concat(balanced_chunks, axis=0).reset_index(drop=True)
                            X = balanced_df.drop(columns=['target_label'])
                            y = balanced_df['target_label']
                            st.sidebar.success("🎉 Balanced via Pure Pandas Undersampling!")
                elif not imbalance_choice.startswith("None") and y is None:
                    st.warning("Imbalance handling requires a Target Column. Skipped.")

                # 7. Feature Selection / Reduction
                if not reduction_choice.startswith("None"):
                    if reduction_choice.startswith("PCA"):
                        # Ensure no NaNs
                        if X.isnull().sum().sum() > 0:
                            st.warning("Cannot apply PCA because there are still missing values. Please impute first.")
                        else:
                            # 1. فصل عمود الـ Target تماماً قبل تشغيل الـ PCA
                            features_to_pca = [c for c in X.columns if c != target_col]
                            
                            if features_to_pca:
                                # تشغيل الـ PCA على الـ Features فقط
                                pca = PCA(n_components=min(5, len(features_to_pca)))
                                pca_features = pca.fit_transform(X[features_to_pca])
                                
                                # تحويل الناتج لـ DataFrame وتسمية الأعمدة PC1, PC2...
                                pca_cols = [f"PC_{i+1}" for i in range(pca_features.shape[1])]
                                X_pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=X.index)
                                
                                # 2. إعادة دمج عمود الـ Target الأصلي مع مكونات الـ PCA بأمان
                                if target_col in X.columns:
                                    X_pca_df[target_col] = X[target_col]
                                    
                                X = X_pca_df # استبدال الجدول القديم بالجدول الجديد المحمي
                            
                    elif reduction_choice.startswith("RFE"):
                        if y is None:
                            st.warning("RFE requires a Target Column. Skipped.")
                        else:
                            # Heuristic to choose estimator based on target type
                            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 20:
                                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                            else:
                                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                                
                            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                            rfe.fit(X, y)
                            selected_features = X.columns[rfe.support_]
                            X = X[selected_features]

                # Combine back X and y
                if target_col != "None" and target_col in processed_df.columns:
                    # Reset indices to ensure safe concatenation, especially after SMOTE/Undersampling
                    X = X.reset_index(drop=True)
                    y = pd.Series(y).reset_index(drop=True)
                    final_df = pd.concat([X, y], axis=1)
                else:
                    final_df = X

                st.session_state['df_preprocessed'] = final_df
                
                st.success("Preprocessing applied successfully with Smart Defaults!")
                st.subheader("Processed Data Preview")
                st.dataframe(final_df.head())
                st.write(f"Shape: {final_df.shape}")
                
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            st.exception(e)

else:
    st.warning("Please upload a dataset on the Upload Data page first.")
