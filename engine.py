import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

@st.cache_resource
def load_screening_model():
    local_file_path = "parkinsons.data"
    
    if not os.path.exists(local_file_path):
        st.error(f"### Critical System Error: '{local_file_path}' missing from your project folder!")
        st.stop()
        
    df = pd.read_csv(local_file_path)
    df.columns = df.columns.str.strip()

    y = df['status'].values
    final_feature_columns = [
    # Voice frequency
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    # Jitter measures
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    # Shimmer measures
    'MDVP:Shimmer', 'Shimmer:APQ3',
    # Noise ratios
    'HNR', 'NHR'
     ]
    X = df[final_feature_columns].values

    # ✅ Split before scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Scaler only sees training data
    scaler_obj = StandardScaler()
    X_train = scaler_obj.fit_transform(X_train)
    X_test = scaler_obj.transform(X_test)

    model_obj = SVC(kernel='rbf', C=10.0,gamma='scale', class_weight='balanced', probability=True, random_state=42)
    model_obj.fit(X_train, y_train)
    
    train_preds = model_obj.predict(X_train)
    test_preds = model_obj.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds) * 100
    test_acc = accuracy_score(y_test, test_preds) * 100
    
    return model_obj, scaler_obj, len(df), train_acc, test_acc
    