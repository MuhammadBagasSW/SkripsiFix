import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#Tambahkan gaya CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f1f8e9;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        font-size: 16px;
        border-radius: 8px;
        padding: 0.5em 1em;
        background-color: #64b5f6;
        color: white;
    }
    .stButton>button:hover {
        background-color: #42a5f5;
   
    }
    /* Ubah sidebar (navbar) */
    section[data-testid="stSidebar"] {
        background-color: #bbdefb !important;
        
    }
    </style>
""", unsafe_allow_html=True)

#Fungsi untuk melakukan normalisasi data
def normalize_data(data, xmin, xmax):
    # Normalisasi data dengan rumus (X - X_min) / (X_max - X_min)
    normalized_data = (data - xmin) / (xmax - xmin)

    return normalized_data


def calculate_nilai_agregat(df):
    # Buat kolom baru Nilai Agregat
    df["NA_Agama"] = (0.6 * df["P_Agama"]) + (0.4 * df["Ket_Agama"])
    df["NA_PPKN"] = (0.6 * df["P_PPKN"]) + (0.4 * df["Ket_PPKN"])
    df["NA_B.Indonesia"] = (0.6 * df["P_B.Indonesia"]) + (0.4 * df["Ket_B.Indonesia"])
    df["NA_Matematika"] = (0.6 * df["P_Matematika"]) + (0.4 * df["Ket_Matematika"])
    df["NA_IPA"] = (0.6 * df["P_IPA"]) + (0.4 * df["Ket_IPA"])
    df["NA_IPS"] = (0.6 * df["P_IPS"]) + (0.4 * df["Ket_IPS"])
    df["NA_Bhs.Inggris"] = (0.6 * df["P_B.Inggris"]) + (0.4 * df["Ket_B.Inggris"])

    return df


import pandas as pd


def encode_nilai(df, kolom_list):
    mapping = {'A': 2, 'B': 1, 'C': 0}
    for kolom in kolom_list:
        encoded_kolom = kolom + '_Encoded'
        df[encoded_kolom] = df[kolom].map(mapping)
    return df

import numpy as np
import pandas as pd
import math
from sklearn.metrics import silhouette_score

def calculate_Dcli_prosedural(normalized_data, radius, squash_factor, index_potensi_tertinggi, accept_ratio, reject_ratio, iteration, M_awal):
    n = len(normalized_data)
    value_indeks_potensi_tertinggi = normalized_data.iloc[index_potensi_tertinggi][numerik_columns]

    Dcl1_results = []
    potensi_akhir = []

    for i in range(n):
        value_indeks_i = normalized_data.iloc[i][numerik_columns]

        DS = {column: (value_indeks_potensi_tertinggi[column] - value_indeks_i[column]) / (radius * squash_factor) for column in numerik_columns}

        sigma_DS_squared = sum(DS[column]**2 for column in DS)
        M_Baru = normalized_data.iloc[i][f'Potensi_tertinggi_{iteration-1}']
        Dcl1 = M_Baru * np.exp(-4) ** sigma_DS_squared

        # Dcl1 for each index (optional print)
        # print(f"Iteration {iteration}, Dcl1 for index {i}: {Dcl1}")

        Dcl1_results.append((i, Dcl1))
        potensi_awal = normalized_data.iloc[i][f'Potensi_Baru_{iteration-1}']
        potensi_akhir_value = potensi_awal - Dcl1

        if potensi_akhir_value < 0:
            potensi_akhir_value = 0

        potensi_akhir.append(potensi_akhir_value)

    normalized_data[f'Potensi_Baru_{iteration}'] = potensi_akhir

    # Update the index_potensi_tertinggi based on the current iteration
    potensi_tertinggi = round(normalized_data[f'Potensi_Baru_{iteration}'].max(), 2)
    index_potensi_tertinggi = normalized_data[f'Potensi_Baru_{iteration}'].idxmax()

    # Calculate M_baru based on the highest potential found
    M_baru = potensi_tertinggi / M_awal

    # Save the results in the DataFrame
    normalized_data[f'Potensi_tertinggi_{iteration}'] = potensi_tertinggi
    normalized_data[f'M_baru_{iteration}'] = M_baru  # Add M_baru column to DataFrame

    # Return the updated M_baru and index_potensi_tertinggi
    return potensi_tertinggi, M_baru, index_potensi_tertinggi

# Example call
# normalized_data = pd.DataFrame(...)  # Pre-normalized DataFrame
# calculate_Dcli(normalized_data, radius, squash_factor, M_awal, index_potensi_tertinggi, accept_ratio, reject_ratio, iteration)

def calculate_multiple_potentials_prosedural(normalized_data, radius, squash_factor, index_potensi_tertinggi,
                                             accept_ratio, reject_ratio, M_awal):
    iteration = 1
    while True:
        potensi_tertinggi, M_baru, index_potensi_tertinggi = calculate_Dcli_prosedural(
            normalized_data, radius, squash_factor, index_potensi_tertinggi, accept_ratio, reject_ratio, iteration,
            M_awal
        )

        # Cek kondisi untuk melanjutkan atau menghentikan iterasi
        if M_baru > accept_ratio:
            indeks_pusat_cluster.append(index_potensi_tertinggi)
            iteration += 1  # Lanjutkan ke iterasi berikutnya

        elif M_baru < accept_ratio and M_baru > reject_ratio:
            indeks_calon_pusat_cluster.append(index_potensi_tertinggi)
            Md = -1
            sd_per_pusat_cluster = cek_md(normalized_data, indeks_pusat_cluster, indeks_calon_pusat_cluster, radius)

            for i, sd in enumerate(sd_per_pusat_cluster, start=1):
                # Kondisi pertama
                if Md < 0 or sd < Md:
                    Md = sd
                else:
                    Md = Md

            # Menghitung Smd (akar dari Md)
            Smd = math.sqrt(Md)
            Rsmd = Smd + M_baru
            # Kondisi penerimaan data sebagai pusat cluster
            if Rsmd >= 1:
                indeks_pusat_cluster.append(index_potensi_tertinggi)
                indeks_calon_pusat_cluster.clear()
                iteration += 1
            elif Rsmd < 1:
                # Update nilai potensi tertinggi menjadi 0
                normalized_data.at[index_potensi_tertinggi, f'Potensi_Baru_{iteration}'] = 0
                indeks_calon_pusat_cluster.clear()
                potensi_tertinggi = round(normalized_data[f'Potensi_Baru_{iteration}'].max(), 2)
                index_potensi_tertinggi = normalized_data[f'Potensi_Baru_{iteration}'].idxmax()
                indeks_calon_pusat_cluster.append(index_potensi_tertinggi)
                Md = -1
                sd_per_pusat_cluster = cek_md(normalized_data, indeks_pusat_cluster, indeks_calon_pusat_cluster, radius)
                for i, sd in enumerate(sd_per_pusat_cluster, start=1):
                    # Kondisi pertama
                    if Md < 0 or sd < Md:
                        Md = sd
                    else:
                        Md = Md

                    # Menghitung Smd (akar dari Md)
                Smd = math.sqrt(Md)
                Rsmd = Smd + M_baru
                M_baru = potensi_tertinggi / M_awal
                if M_baru < reject_ratio:
                    Rsmd = M_baru
                    indeks_calon_pusat_cluster.clear()
                    return Rsmd
                    break

                normalized_data[f'M_baru_{iteration}'] = M_baru
                if Rsmd >= 1:
                    iteration += 1
                if Rsmd < 1:
                    # Update nilai potensi tertinggi menjadi 0
                    normalized_data.at[index_potensi_tertinggi, f'Potensi_Baru_{iteration}'] = 0
                    indeks_calon_pusat_cluster.clear()

                    potensi_tertinggi = round(normalized_data[f'Potensi_Baru_{iteration}'].max(), 2)
                    index_potensi_tertinggi = normalized_data[f'Potensi_Baru_{iteration}'].idxmax()
                    M_baru = potensi_tertinggi / M_awal
                    normalized_data[f'M_baru_{iteration}'] = M_baru
                    indeks_calon_pusat_cluster.append(index_potensi_tertinggi)
                    Md = -1
                    while True:  # Mulai loop untuk menghitung ulang Rsmd
                        if M_baru < reject_ratio:
                            Rsmd = M_baru
                            indeks_calon_pusat_cluster.clear()
                            return Rsmd
                            break
                        sd_per_pusat_cluster = cek_md(normalized_data, indeks_pusat_cluster, indeks_calon_pusat_cluster,
                                                      radius)
                        for i, sd in enumerate(sd_per_pusat_cluster, start=1):
                            # Kondisi pertama
                            if Md < 0 or sd < Md:
                                Md = sd
                            else:
                                Md = Md

                        # Menghitung Smd (akar dari Md)
                        Smd = math.sqrt(Md)
                        Rsmd = Smd + M_baru

                        # Jika Rsmd >= 1, keluar dari loop
                        if Rsmd >= 1:
                            indeks_pusat_cluster.append(index_potensi_tertinggi)
                            indeks_calon_pusat_cluster.clear()
                            iteration += 1
                            break  # Keluar dari while loop karena Rsmd sudah memenuhi syarat
                        if Rsmd < 1 and M_baru > reject_ratio:
                            normalized_data.at[index_potensi_tertinggi, f'Potensi_Baru_{iteration}'] = 0
                            indeks_calon_pusat_cluster.clear()
                            potensi_tertinggi = round(normalized_data[f'Potensi_Baru_{iteration}'].max(), 2)
                            index_potensi_tertinggi = normalized_data[f'Potensi_Baru_{iteration}'].idxmax()
                            M_baru = potensi_tertinggi / M_awal
                            normalized_data[f'M_baru_{iteration}'] = M_baru
                            indeks_calon_pusat_cluster.append(index_potensi_tertinggi)
                            Md = -1

        else:
            print("Perhitungan Selesai Dilakukan dan tidak ada calon pusat cluster lagi")
            break

    return normalized_data

def calculate_potential(normalized_data, radius):
    dist_kj(normalized_data, radius)  # Jika ada fungsi dist_kj, pastikan fungsinya didefinisikan sebelumnya.

    # Membuat list kosong untuk menampung hasil
    D = []

    # Loop untuk baris 1 sampai n
    for i in range(len(df)):  # Looping n kali
        result = 0  # Variabel untuk menjumlahkan hasil DS1, DS2, ..., DSn untuk setiap baris
        for j in range(1, df.shape[0] + 1):  # Looping untuk kolom DS1 sampai DSn
            result += np.exp(-4) ** normalized_data[f"DS{j}"].iloc[i]  # Menghitung np.exp(-4) ** DS untuk setiap kolom DS
        D.append(result)  # Menambahkan hasil per baris ke dalam list D

    # Menambahkan hasil perhitungan D ke dalam DataFrame sebagai kolom baru
    normalized_data['Potensi_Baru_0'] = D

    # Menghitung jumlah total dari hasil D
    total_sum_D = sum(D)
    potensi_tertinggi = normalized_data['Potensi_Baru_0'].max()
    indeks_tertinggi = normalized_data['Potensi_Baru_0'].idxmax()
     # Save the results in the DataFrame
    normalized_data['Potensi_tertinggi_0'] = potensi_tertinggi
    # Menampilkan potensi tertinggi dan indeksnya
    print(f"Potensi awal titik tertinggi berada pada indeks ke-{indeks_tertinggi} dengan potensi {potensi_tertinggi}")

    # Mengembalikan nilai potensi tertinggi dan indeks tertinggi
    return potensi_tertinggi, indeks_tertinggi

import warnings
def dist_kj(normalized_data, radius):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    n, m = normalized_data.shape
    columns = normalized_data.columns  # Ambil semua kolom dari data secara otomatis

    for i in range(n):  # Loop untuk setiap baris
        dist_columns = []  # Reset list dist_columns untuk setiap baris i
        for column in columns:  # Loop untuk setiap kolom
            # Menghitung nilai Dist berdasarkan rumus yang diberikan
            dist_value = ((normalized_data[column].iloc[i] - normalized_data[column]) / radius) ** 2
            dist_columns.append(dist_value)

        # Menghitung DS untu/k setiap baris dan menyimpannya
        normalized_data[f'DS{i+1}'] = sum(dist_columns)
    return normalized_data

def cek_md(normalized_data, indeks_pusat_cluster, indeks_calon_pusat_cluster, radius):
    # Menentukan jumlah pusat cluster yang ada dan calon pusat cluster
    n = len(indeks_pusat_cluster)
    m = len(indeks_calon_pusat_cluster)


    value_indeks_pusat_cluster = normalized_data.iloc[indeks_pusat_cluster][numerik_columns]

    # Membuat list untuk menyimpan hasil SD per pusat cluster
    sd_per_pusat_cluster = []

    # Mengurangi elemen dari indeks_calon_pusat_cluster sebanyak n
    for i in range(n):
        # Pilih elemen calon pusat cluster berdasarkan iterasi yang dikurangi
        if i < n:  # Pastikan indeksnya valid
            value_indeks_i = normalized_data.iloc[indeks_calon_pusat_cluster[m-1]][numerik_columns]

            # Hitung SD per kolom berdasarkan calon pusat cluster
            Sd_i = {column: (value_indeks_pusat_cluster.iloc[i][column] - value_indeks_i[column]) / radius for column in numerik_columns}

            # Hitung sigma DS squared untuk calon pusat cluster saat ini
            sigma_DS_squared_i = sum(Sd_i[column]**2 for column in Sd_i)

            # Simpan hasil SD per pusat cluster
            sd_per_pusat_cluster.append(sigma_DS_squared_i)
    return sd_per_pusat_cluster

import math

def calculate_sigma(xmin, xmax, radius):
    # Inisialisasi list untuk menyimpan hasil sigma_rj


    # Perhitungan sigma_rj untuk setiap pasangan nilai xmin dan xmax
    for i in range(len(xmin)):
        # Perhitungan sigma_rj
        sigma_rj = radius * (xmin[i] - xmax[i]) / math.sqrt(8)

        # Mengambil nilai mutlak dari sigma_rj dan memastikan hasilnya tipe float
        sigma_rj_abs = float(abs(sigma_rj))

        # Menyimpan hasil mutlak sigma_rj dalam list sigma
        sigma.append(sigma_rj_abs)

    return sigma  # Mengembalikan list sigma yang berisi hasil perhitungan


class evaluate_silhouette:
    def __init__(self, df, df_pusat_fix, df_sigma, features):
        self.df = df
        self.df_pusat_fix = df_pusat_fix
        self.df_sigma = df_sigma
        self.features = features
        self.df_derajat = None
        self.df_final = None

    def calculate_membership(self, format_degrees=True, decimal_places=6):
        df_derajat = pd.DataFrame()
        K = len(self.df_pusat_fix)

        for k in range(K):
            mu_columns = []
            for i in range(len(self.df)):
                diff = self.df.iloc[i][self.features] - self.df_pusat_fix.iloc[k][self.features]
                scaled_diff = diff / (math.sqrt(2) * self.df_sigma.iloc[0][self.features])
                sum_squared = (scaled_diff ** 2).sum()
                mu_columns.append(sum_squared)
            df_derajat[f'derajat_cluster_{k}'] = np.exp(-np.array(mu_columns))

        cluster_columns = [f'derajat_cluster_{k}' for k in range(K)]
        df_derajat['cluster'] = np.argmax(df_derajat[cluster_columns].values, axis=1)

        if format_degrees:
            for col in cluster_columns:
                df_derajat[col] = df_derajat[col].apply(lambda x: f"{x:.{decimal_places}f}")

        self.df_derajat = df_derajat
        self.df_final = pd.concat([self.df, df_derajat['cluster']], axis=1)
        return self.df_final, self.df_derajat

    def evaluate_silhouette(self, xmin, xmax, normalize=True, verbose=True):
        if self.df_final is None:
            raise ValueError("Harap jalankan calculate_membership() terlebih dahulu.")

        X = self.df_final[self.features].values
        y = self.df_final['cluster'].values

        if normalize:
            X = self._normalize(X, xmin, xmax)

        score = silhouette_score(X, y)

        if verbose:
            st.write(f"Hasil dari Silhouette Score adalah: {score:.4f}")

        return score

    def _normalize(self, data, xmin, xmax):
        return (data - xmin) / (xmax - xmin)


def FSC_model(data, xmin, xmax, radius, squash_factor, accept_ratio, reject_ratio, ):
    normalized_data = normalize_data(df, xmin, xmax)
    n, m = normalized_data.shape
    potensi_tertinggi, indeks_tertinggi = calculate_potential(normalized_data, radius)
    M_awal = normalized_data.iloc[1][f'Potensi_tertinggi_{0}']
    index_potensi_tertinggi = indeks_tertinggi
    indeks_pusat_cluster.append(indeks_tertinggi)  # Pastikan nilai pertama sudah ditambahkan ke dalam list
    # Pastikan variabel digunakan dalam pemanggilan fungsi lainnya
    calculate_multiple_potentials_prosedural(normalized_data, radius, squash_factor, index_potensi_tertinggi,
                                             accept_ratio, reject_ratio, M_awal)
    jumlah_data_cluster = len(df.iloc[indeks_pusat_cluster])
    # Memanggil fungsi dan menampilkan hasil
    sigma = calculate_sigma(xmin, xmax, radius)
    df_pusat_fix = df.iloc[indeks_pusat_cluster, 0:13]

    df_sigma = pd.DataFrame([sigma], columns=columns)
    # Inisialisasi model
    model = evaluate_silhouette(
        df=df,
        df_pusat_fix=df_pusat_fix,
        df_sigma=df_sigma,
        features=columns
    )

    # Hitung derajat keanggotaan dan kluster
    df_final, df_derajat = model.calculate_membership()
    df_derajat.rename(columns={'derajat_cluster_5': 'derajat_cluster_6'}, inplace=True)
    df_derajat.rename(columns={'derajat_cluster_4': 'derajat_cluster_5'}, inplace=True)
    df_derajat.rename(columns={'derajat_cluster_3': 'derajat_cluster_4'}, inplace=True)
    df_derajat.rename(columns={'derajat_cluster_2': 'derajat_cluster_3'}, inplace=True)
    df_derajat.rename(columns={'derajat_cluster_1': 'derajat_cluster_2'}, inplace=True)
    df_derajat.rename(columns={'derajat_cluster_0': 'derajat_cluster_1'}, inplace=True)

    df_derajat['cluster'] = df_derajat['cluster'].replace({
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
    })
    df_final['cluster'] = df_final['cluster'].replace({
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
    })
    st.write("")
    st.write("")
    st.header("Hasil Pusat Cluser")
    st.write(f"Jumlah cluster yang terbentuk = {jumlah_data_cluster}")
    st.write("Pusat clusternya adalah data berikut:\n")
    st.write(df_pusat_fix)
    st.write("")
    st.write("")
    st.header("Hasil Derajat Keanggotaan")
    st.dataframe(df_derajat)
    st.write("")
    st.write("")
    st.header("Hasil Akhir Cluster")
    st.dataframe(df_final)
    st.write("")
    st.write("")
    st.header("Hasil Evaluasi Model")
    score = model.evaluate_silhouette(xmin, xmax)
    st.write("")
    st.write("")
    st.header("Hasil Profil Setiap Cluster")
    st.dataframe(df_final.describe())

    # Simpan hasil akhir ke session state
    st.session_state['df_final'] = df_final





















# Variabel numerik & normalisasi
numerik_columns = [
    'NA_Agama', 'NA_PPKN', 'NA_B.Indonesia', 'NA_Matematika', 'NA_IPA',
    'NA_IPS', 'NA_Bhs.Inggris', 'Nilai_Sosial_Encoded', 'Nilai_Spiritual_Encoded'
]
xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
xmax = np.array([100, 96.2, 99.2, 98, 97.2, 96, 96, 2, 2])

# Sidebar navigasi
st.sidebar.title("📁 Menu Navigasi")
menu = st.sidebar.radio("Berikut adalah fitur yang tersedia: ", ["Input Data 📥","Visualisasi 📊", "Clustering 🔍"])

# ======================
# CRUD DATA PAGE
# ======================
# CRUD DATA PAGE
# ======================
if menu == "Input Data 📥":
    st.title("📥 Upload Data Siswa")

    uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            df = calculate_nilai_agregat(df)
            df = encode_nilai(df, ['Nilai_Sosial', 'Nilai_Spiritual'])
            st.session_state['df'] = df
            st.success("Data berhasil diunggah dan disimpan!")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

    # Tampilkan data jika sudah ada
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("📊 Data Saat Ini")
        df_filtered = df.drop(columns=numerik_columns)
        st.dataframe(df_filtered)

        # Tombol untuk menghapus semua data
        if st.button("🗑️ Hapus Semua Data", help="Klik untuk menghapus seluruh data yang telah diunggah."):
            del st.session_state['df']
            st.warning("Semua data telah dihapus.")
            st.rerun()

elif menu == "Visualisasi 📊":
    st.title("Visualisasi Data")
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu melalui menu 'Input Data'.")
        st.stop()
    df = st.session_state['df']
    if df is None or df.empty:
        st.error("Data kosong. Silakan upload ulang di menu 'Input Data'.")
        st.stop()
    st.write("")  # baris kosong
    # ===== FILTER INTERAKTIF (Kelas) =====
    with st.expander("🎚️ Filter Data"):
        col1, col2 = st.columns(2)

        with col1:
            kelas_tersedia = df['Kelas'].unique().tolist() if 'Kelas' in df.columns else []
            selected_kelas = st.multiselect("Filter berdasarkan Kelas", kelas_tersedia, default=kelas_tersedia)

        # Terapkan filter
        if 'Kelas' in df.columns:
            df = df[df['Kelas'].isin(selected_kelas)]


    # ===== TAMPILKAN DATA =====
    st.subheader("📄 Data yang Tersedia")
    st.dataframe(df, use_container_width=True)

    # ===== STATISTIK DESKRIPTIF =====
    with st.expander("📈 Statistik Deskriptif"):
        st.dataframe(df.describe(include='all').T, use_container_width=True)

    # ===== VISUALISASI INTERAKTIF =====
    st.markdown("---")
    st.subheader("📊 Visualisasi Interaktif")

    # Pilih kolom numerik
    numerik_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    col3, col4 = st.columns(2)
    with col3:
        x_var = st.selectbox("Variabel X (Scatter)", numerik_cols, index=0)
    with col4:
        y_var = st.selectbox("Variabel Y (Scatter)", numerik_cols, index=1 if len(numerik_cols) > 1 else 0)

    # Scatter plot interaktif
    fig_scatter = px.scatter(
        df, x=x_var, y=y_var, color=df['Kelas'] if 'Kelas' in df.columns else None,
        hover_data=df.columns, title=f"🔹 Scatter Plot: {x_var} vs {y_var}",
        template='plotly_white', height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ===== PIE CHART KOMPOSISI KELAS =====
    if 'Kelas' in df.columns:
        st.subheader("🧁 Komposisi Jumlah Siswa per Kelas")
        kelas_counts = df['Kelas'].value_counts().reset_index()
        kelas_counts.columns = ['Kelas', 'Jumlah']
        fig_pie = px.pie(kelas_counts, names='Kelas', values='Jumlah',
                         title="Distribusi Siswa per Kelas", hole=0.4, template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)

    # ===== HISTOGRAM NILAI AKADEMIK =====
    st.subheader("📚 Distribusi Nilai Akademik")
    selected_hist = st.selectbox("Pilih Nilai Akademik", numerik_cols, key="histogram")
    fig_hist = px.histogram(df, x=selected_hist, nbins=20,
                            title=f"Distribusi {selected_hist}", template='plotly_white')
    st.plotly_chart(fig_hist, use_container_width=True)



# ======================
# CLUSTERING PAGE
# ======================
elif menu == "Clustering 🔍":
    st.title("🔍 Fuzzy Subtractive Clustering")

    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu melalui menu 'Input Data'.")
        st.stop()

    df = st.session_state['df']
    if df is None or df.empty:
        st.error("Data kosong. Silakan upload ulang di menu 'Input Data'.")
        st.stop()

    tab1, tab2 = st.tabs(["🔧 Parameter & Proses Clustering", "📊 Visualisasi Hasil Clustering"])

    with tab1:
        # ✅ Tampilkan semua data hasil upload
        st.write("")  # baris kosong
        st.subheader("\n📄 Data Yang Tersedia")
        st.write("")
        st.dataframe(df)
        st.write("")

        # ✅ Buat dan tampilkan data untuk clustering saja
        df_clustering = df[numerik_columns]

        st.markdown("### ⚙️ Parameter FSC (Fuzzy Subtractive Clustering)")
        radius = st.number_input("Radius", min_value=0.18, max_value=0.3, value=0.3, step=0.05, format="%.2f")
        squash_factor = st.number_input("Squash Factor", min_value=0.53, max_value=1.35, value=1.25, step=0.05, format="%.2f")
        accept_ratio = st.number_input("Accept Ratio", min_value=0.3, max_value=0.6, value=0.5, step=0.05, format="%.2f")
        reject_ratio = st.number_input("Reject Ratio", min_value=0.03, max_value=0.2, value=0.04, step=0.01, format="%.2f")

        # Validasi agar accept_ratio tidak lebih rendah dari reject_ratio
        if accept_ratio < reject_ratio:
            st.error("❌ Accept Ratio tidak boleh lebih rendah dari Reject Ratio!")

        if st.button("Mulai Cluster"):
            with st.spinner("⏳ Sedang menjalankan FSC..."):


                indeks_pusat_cluster = []  # List kosong untuk menyimpan indeks pusat cluster
                indeks_calon_pusat_cluster = []
                rasio = []
                sigma = []

                # Pilih variabel yang akan digunakan
                numerik_columns = [
                    'NA_Agama', 'NA_PPKN', 'NA_B.Indonesia', 'NA_Matematika', 'NA_IPA',
                    'NA_IPS', 'NA_Bhs.Inggris',
                    'Nilai_Sosial_Encoded', 'Nilai_Spiritual_Encoded',
                ]
                selected_columns = [
                    'NA_Agama', 'NA_PPKN', 'NA_B.Indonesia', 'NA_Matematika',
                    'NA_IPA', 'NA_IPS', 'NA_Bhs.Inggris'
                ]
                # Definisikan nilai X_min dan X_max
                xmin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
                xmax = np.array([100, 96.2, 99.2, 98, 97.2, 96, 96, 2, 2])

                df = calculate_nilai_agregat(df)
                st.header("Hasil Pra Pemrosesan Data")
                st.dataframe(df[selected_columns])

                df = encode_nilai(df, ['Nilai_Sosial', 'Nilai_Spiritual'])
                st.header("Hasil Label Encoding Data")
                st.dataframe(df[numerik_columns])

                features = numerik_columns
                columns = numerik_columns
                df = df[columns]

                # Jalankan clustering
                FSC_model(df, xmin, xmax, radius=radius, squash_factor=squash_factor,
                          accept_ratio=accept_ratio, reject_ratio=reject_ratio)
            st.success("✅ Clustering selesai!")
            if st.button("🔁 Rerun Clustering", key="rerun_button"):
                if 'df_final' in st.session_state:
                    del st.session_state['df_final']
                st.experimental_rerun()

    with tab2:



        if 'df_final' not in st.session_state:
            st.info("ℹ️ Silakan jalankan proses clustering terlebih dahulu.")
        else:
            df_final = st.session_state['df_final']
            st.subheader("📊 Visualisasi Terbaru Hasil Clustering")

            # Contoh kriteria kelayakan (bisa disesuaikan)
            layak_lolos = (df_final is not None) and (not df_final.empty)

            if layak_lolos:
                if st.button("📈 Visualisasi"):
                    with st.spinner("⏳ Sedang Menampilkan Visualisasi..."):

                        # Ambil kolom 'NAMA' dan 'Kelas' dari df
                        df_filtered = df[["NAMA", "Kelas"]]

                        # Gabungkan dengan df_final secara baris sejajar
                        df_final = pd.concat([df_filtered.reset_index(drop=True), df_final.reset_index(drop=True)],
                                                      axis=1)

                        # Tampilkan di Streamlit
                        st.markdown("## 📋 Data Hasil Clustering")
                        st.dataframe(df_final)

                        # Benar: Jumlah 7 mata pelajaran / 7
                        df_final["Rata_Rata_Nilai"] = (df["NA_Agama"] + df["NA_PPKN"] + df["NA_B.Indonesia"] +
                                                       df["NA_Matematika"] + df["NA_IPA"] + df["NA_IPS"] +
                                                       df["NA_Bhs.Inggris"]) / 7

                        df_final['cluster'] = df_final['cluster'].astype(str)
                        colors = px.colors.qualitative.Set1
                        unique_clusters = df_final['cluster'].unique()
                        data = []

                        for i, cluster_id in enumerate(unique_clusters):
                            cluster_df = df_final[df_final['cluster'] == cluster_id]
                            trace = go.Scatter3d(
                                x=cluster_df['Rata_Rata_Nilai'],
                                y=cluster_df['Nilai_Sosial_Encoded'],
                                z=cluster_df['Nilai_Spiritual_Encoded'],
                                mode='markers',
                                name=f'Cluster {cluster_id}',
                                marker=dict(
                                    size=5,
                                    color=colors[i % len(colors)]
                                ),
                                text=cluster_df['cluster']
                            )
                            data.append(trace)
                        st.header("3D Visualisasi Hasil Cluster")
                        layout = go.Layout(
                            scene=dict(
                                xaxis=dict(title='Rata-Rata Nilai Pelajaran'),
                                yaxis=dict(title='Nilai Sosial'),
                                zaxis=dict(title='Nilai Spiritual')
                            ),
                            margin=dict(l=0, r=0, b=0, t=50),
                            width=1000,
                            height=700,
                            legend=dict(
                                x=0,
                                y=1,
                                font=dict(size=12),
                                bgcolor='rgba(255,255,255,0.7)',
                                bordercolor='black',
                                borderwidth=1
                            )
                        )

                        fig = go.Figure(data=data, layout=layout)
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("")
                        st.write("")
                        st.header("Pairplot Visualisasi Hasil Cluster")
                        # Buat pairplot
                        pairplot = sns.pairplot(
                            df_final,
                            hue='cluster',
                            vars=["Rata_Rata_Nilai", "Nilai_Sosial_Encoded", "Nilai_Spiritual_Encoded"],
                            diag_kind='hist',
                            plot_kws={'alpha': 0.5, 's': 60, 'edgecolor': 'w'},
                            height=5,
                            aspect=1.2
                        )

                        # Tambahkan judul
                        plt.suptitle("Pairplot Visualization Cluster", y=1.02, fontsize=16)

                        # Tampilkan di Streamlit
                        st.pyplot(pairplot.figure)

                        st.write("")
                        st.write("")
                        st.header("Pairplot Visualisasi Hasil Cluster")
                        import plotly.express as px

                        unique_clusters = sorted(df_final['cluster'].unique())
                        color_palette = px.colors.qualitative.Set1
                        cluster_color_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in
                                             enumerate(unique_clusters)}

                        st.header("📊 Distribusi Cluster Berdasarkan Kelas (Pie Chart)")

                        # Ambil daftar kelas unik
                        kelas_list = sorted(df_final['Kelas'].unique())

                        for kelas in kelas_list:
                            st.subheader(f"Kelas {kelas}")

                            # Filter data berdasarkan kelas
                            df_kelas = df_final[df_final['Kelas'] == kelas]

                            if df_kelas.empty:
                                st.info(f"Tidak ada data untuk Kelas {kelas}")
                                continue

                            # Hitung jumlah anggota per cluster
                            cluster_counts = df_kelas['cluster'].value_counts().reset_index()
                            cluster_counts.columns = ['cluster', 'jumlah']
                            cluster_counts['persen'] = (cluster_counts['jumlah'] / cluster_counts['jumlah'].sum()) * 100

                            # Buat pie chart
                            fig = px.pie(
                                cluster_counts,
                                names='cluster',
                                values='jumlah',
                                color='cluster',
                                color_discrete_map=cluster_color_map,  # Gunakan warna yang konsisten
                                hole=0.3,
                                title=f"Distribusi Cluster di Kelas {kelas}",
                            )
                            fig.update_traces(textinfo='percent+label')

                            # Tampilkan chart di Streamlit
                            st.plotly_chart(fig, use_container_width=True)



            else:
                st.warning("Data cluster tidak valid atau kosong, tidak bisa visualisasi.")
