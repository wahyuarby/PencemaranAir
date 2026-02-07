import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

# Set konfigurasi halaman
st.set_page_config(layout="wide", page_title="Aplikasi Data Mining Potabilitas Air 💧")

# --- Judul Aplikasi ---
st.title("Aplikasi Data Mining Potabilitas Air 💧")
st.write("""
Aplikasi ini memungkinkan Anda untuk melihat ringkasan dataset potabilitas air dan memprediksi apakah air tersebut layak minum atau tidak
menggunakan model Machine Learning.
""")

# --- Sidebar Navigasi ---
st.sidebar.header("Navigasi Aplikasi")
analysis_type = st.sidebar.radio(
    "Pilih Halaman:",
    ("Beranda (Home)", "Prediksi Potabilitas Air")
)

# --- Fungsi Muat Data (Cache Data) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("water_potability_balanced.csv")
    except FileNotFoundError:
        st.error("Pastikan file 'water_potability_balanced.csv' berada di direktori yang sama dengan aplikasi.")
        st.stop()
    return df

# --- Fungsi Pelatihan Model (Cache Resource) ---
# Ini mencegah training ulang setiap kali tombol diklik
@st.cache_resource
def train_model(df):
    # Preprocessing
    missing_values_before = df.isnull().sum()
    if missing_values_before.any():
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        df_imputed = df.copy()
    
    df_imputed['Potability'] = df_imputed['Potability'].astype(int)
    
    selected_features = ['Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Organic_carbon']
    X = df_imputed[selected_features]
    y = df_imputed['Potability']
    
    # Split Data 90:10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    
    # Return model, data imputed (untuk range slider), dan metrik statis
    return model, df_imputed, selected_features

# --- Load Data Utama ---
df = load_data()

# --- Load Model & Data (Hanya dijalankan sekali) ---
model, df_imputed, selected_features = train_model(df)

# ==========================================
# BAGIAN 1: BERANDA (HOME)
# ==========================================
if analysis_type == "Beranda (Home)":
    st.header("Beranda: Gambaran Umum Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Cuplikan Data (5 Teratas & 5 Terbawah)")
        preview_df = pd.concat([df.head(), df.tail()])
        st.dataframe(preview_df, use_container_width=True)
        st.write(f"**Dimensi Data:** {df.shape[0]} Baris, {df.shape[1]} Kolom")

    with col2:
        st.info("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    
    st.subheader("Proporsi Data Target (Potabilitas)")
    potability_counts = df_imputed['Potability'].value_counts().rename(index={0: 'Tidak Layak Minum', 1: 'Layak Minum'})
    
    fig_pie = px.pie(
        values=potability_counts.values,
        names=potability_counts.index,
        title='Distribusi Potabilitas Air',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# BAGIAN 2: PREDIKSI POTABILITAS AIR
# ==========================================
elif analysis_type == "Prediksi Potabilitas Air":
    st.header("Prediksi Potabilitas Air")
    
    # --- Tampilkan Metrik Performansi (LANGSUNG DITAMPILKAN DI ATAS) ---
    st.info("Model Random Forest (Split 90:10) siap digunakan.")
    st.subheader("Performa Model")
    
    # Metrik Statis sesuai permintaan
    accuracy = 0.8550
    precision = 0.8585
    recall = 0.8500
    f1 = 0.8542
    roc_auc = 0.9073

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Akurasi", f"{accuracy:.2%}")
    col2.metric("Presisi", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")
    col5.metric("ROC-AUC", f"{roc_auc:.2%}")

    st.markdown("---")

    # --- Form Input User ---
    st.subheader("📝 Masukkan Parameter Air")
    
    input_data = {}
    col_input1, col_input2 = st.columns(2)
    
    features_list = list(selected_features)
    half = len(features_list) // 2
    
    with col_input1:
        for column in features_list[:half]:
            min_val = float(df_imputed[column].min())
            max_val = float(df_imputed[column].max())
            mean_val = float(df_imputed[column].mean())
            step = 1.0 if (max_val - min_val) > 10 else 0.01
            
            input_data[column] = st.number_input(
                f"{column} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val, max_value=max_val, value=mean_val, step=step,
                key=f"in_{column}" # Key unik agar tidak reset
            )

    with col_input2:
        for column in features_list[half:]:
            min_val = float(df_imputed[column].min())
            max_val = float(df_imputed[column].max())
            mean_val = float(df_imputed[column].mean())
            step = 1.0 if (max_val - min_val) > 10 else 0.01
            
            input_data[column] = st.number_input(
                f"{column} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val, max_value=max_val, value=mean_val, step=step,
                key=f"in_{column}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Prediksi
    predict_button = st.button("🔍 Prediksi Sekarang", use_container_width=True)

    # --- Hasil Prediksi (Hanya muncul setelah klik) ---
    if predict_button:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        # Ensure correct column order
        input_df = input_df[selected_features]

        # Predict using cached model
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.markdown("---")
        st.subheader("Hasil Analisis:")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            if prediction[0] == 1:
                st.success("### ✅ LAYAK MINUM")
                st.image("https://img.icons8.com/color/96/000000/water-glass.png", caption="Aman Dikonsumsi")
            else:
                st.error("### 🚫 TIDAK LAYAK")
                st.image("https://img.icons8.com/color/96/000000/biohazard.png", caption="Berbahaya")

        with col_res2:
            st.write("#### Probabilitas Kepercayaan Model:")
            st.progress(prediction_proba[0][1])
            st.write(f"Kemungkinan Layak Minum: **{prediction_proba[0][1]*100:.2f}%**")
            st.write(f"Kemungkinan Tidak Layak: **{prediction_proba[0][0]*100:.2f}%**")
