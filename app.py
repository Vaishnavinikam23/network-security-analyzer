import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
@st.cache_data
def load_and_train_model():
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    selected_features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "logged_in"]
    label_column = "label"

    # Load datasets
    train_df = pd.read_csv("KDDTrain+.txt", names=column_names)
    test_df = pd.read_csv("KDDTest+.txt", names=column_names)

    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"], inplace=True)

    train_df = train_df[selected_features + [label_column]]
    test_df = test_df[selected_features + [label_column]]

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    # Label: normal = 0, attack = 1
    train_df[label_column] = train_df[label_column].apply(lambda x: 0 if x == "normal" else 1)
    test_df[label_column] = test_df[label_column].apply(lambda x: 0 if x == "normal" else 1)

    # Prepare training data
    X_train = train_df.drop(label_column, axis=1)
    y_train = train_df[label_column]
    X_test = test_df.drop(label_column, axis=1)
    y_test = test_df[label_column]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

    return model, scaler, encoders, selected_features

# Load model
model, scaler, encoders, selected_features = load_and_train_model()

# UI
st.title("🔐 Network Intrusion Detection")
st.markdown("Predict if a network connection is **normal** or an **attack**")

# User Input Form
with st.form("user_input_form"):
    duration = st.number_input("Duration (in seconds)", min_value=0.0, value=0.0)
    protocol_type = st.selectbox("Protocol Type", options=encoders["protocol_type"].classes_)
    service = st.selectbox("Service", options=encoders["service"].classes_)
    flag = st.selectbox("Flag", options=encoders["flag"].classes_)
    src_bytes = st.number_input("Bytes sent from source", min_value=0.0, value=0.0)
    dst_bytes = st.number_input("Bytes received by destination", min_value=0.0, value=0.0)
    logged_in = st.radio("User Logged In?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    input_dict = {
        "duration": duration,
        "protocol_type": encoders["protocol_type"].transform([protocol_type])[0],
        "service": encoders["service"].transform([service])[0],
        "flag": encoders["flag"].transform([flag])[0],
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "logged_in": logged_in
    }

    df_input = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(df_input)
    pred_prob = model.predict(scaled_input)[0][0]
    prediction = 1 if pred_prob > 0.5 else 0

    st.subheader("🔎 Result:")
    if prediction == 1:
        st.error("⚠️ **Attack Detected!**")
    else:
        st.success("✅ **Normal Traffic**")

    st.caption(f"Confidence Score: `{pred_prob:.4f}`")
