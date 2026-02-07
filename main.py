import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Set the title of the Streamlit application
st.set_page_config(layout="wide", page_title="Aplikasi Data Mining Potabilitas Air 💧")

st.title("Aplikasi Data Mining Potabilitas Air 💧")
st.write("""
Aplikasi ini memungkinkan Anda untuk melihat ringkasan dataset potabilitas air dan memprediksi apakah air tersebut layak minum atau tidak
menggunakan model Machine Learning.
""")

# --- Bagian Sidebar untuk Navigasi ---
st.sidebar.header("Navigasi Aplikasi")
# HANYA MENAMPILKAN 2 PILIHAN: HOME DAN PREDIKSI
analysis_type = st.sidebar.radio(
    "Pilih Halaman:",
    ("Beranda (Home)", "Prediksi Potabilitas Air")
)

# --- Muat Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("water_potability_balanced.csv")
    except FileNotFoundError:
        st.error("Pastikan file 'water_potability_balanced.csv' berada di direktori yang sama dengan aplikasi.")
        st.stop()
    return df

df = load_data()

# --- Preprocessing Data Awal (Imputasi Missing Values) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Status Data")
missing_values_before = df.isnull().sum()
if missing_values_before.any():
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
else:
    df_imputed = df.copy()

# Ensure 'Potability' column is integer type
df_imputed['Potability'] = df_imputed['Potability'].astype(int)

# --- Definisi Fitur yang Digunakan untuk Model ---
selected_features = ['Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Organic_carbon']

# Split features (X) and target (y)
X = df_imputed[selected_features]
y = df_imputed['Potability']

# ==========================================
# BAGIAN 1: BERANDA (HOME / GAMBARAN UMUM)
# ==========================================
if analysis_type == "Beranda (Home)":
    st.header("Beranda: Gambaran Umum Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("5 Baris Pertama Data")
        st.dataframe(df.head(2367), use_container_width=True)
        
        st.write(f"**Dimensi Data:** {df.shape[0]} Baris, {df.shape[1]} Kolom")

    with col2:
        st.info("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")if analysis_type == "Beranda (Home)":
    st.header("Beranda: Gambaran Umum Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Cuplikan Data (5 Teratas & 5 Terbawah)")
        
        # Menggabungkan 5 baris pertama dan 5 baris terakhir
        preview_df = pd.concat([df.head(), df.tail()])
        
        # Menampilkan dataframe
        st.dataframe(preview_df, use_container_width=True)
        
        st.write(f"**Dimensi Data:** {df.shape[0]} Baris, {df.shape[1]} Kolom")

    with col2:
        st.info("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    
    # Visualisasi Distribusi Kelas (Pie Chart)
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
    st.write("Model Machine Learning (Random Forest) akan dilatih secara real-time untuk memprediksi kualitas air.")

    # --- Training Model ---
    with st.spinner('Sedang melatih model...'):
        # Split Data (Default settings for simplicity, hidden from user to make UI cleaner, 
        # or you can keep sliders if you want advanced control)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Initialize and Train Model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

    # --- Tampilkan Metrik Performansi Model ---
    st.subheader("Performa Model Saat Ini")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy:.2%}")
    col2.metric("Presisi", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    st.markdown("---")

    # --- Form Input User ---
    st.subheader("📝 Masukkan Parameter Air")
    
    # Membuat form input dalam 2 kolom agar lebih rapi
    input_data = {}
    col_input1, col_input2 = st.columns(2)
    
    features_list = list(selected_features)
    half = len(features_list) // 2
    
    # Kolom Kiri
    with col_input1:
        for column in features_list[:half]:
            min_val = float(df_imputed[column].min())
            max_val = float(df_imputed[column].max())
            mean_val = float(df_imputed[column].mean())
            
            # Logic step slider
            step = 1.0 if (max_val - min_val) > 10 else 0.01
            
            input_data[column] = st.number_input(
                f"{column} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val, max_value=max_val, value=mean_val, step=step
            )

    # Kolom Kanan
    with col_input2:
        for column in features_list[half:]:
            min_val = float(df_imputed[column].min())
            max_val = float(df_imputed[column].max())
            mean_val = float(df_imputed[column].mean())
            
            step = 1.0 if (max_val - min_val) > 10 else 0.01
            
            input_data[column] = st.number_input(
                f"{column} (Range: {min_val:.1f} - {max_val:.1f})",
                min_value=min_val, max_value=max_val, value=mean_val, step=step
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Prediksi Sekarang", use_container_width=True)

    # --- Hasil Prediksi ---
    if predict_button:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        # Ensure correct column order
        input_df = input_df[selected_features]

        # Predict
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

