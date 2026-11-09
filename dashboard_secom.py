"""SECOM Insight | Dashboard pembelajaran berurutan, mengikuti hasil Google Colab (Bagian 1–28).

Pengolahan data untuk EDA: dataset dosen yang identik dengan notebook.
Metrik model: dibaca dari output notebook yang telah disimpan di results/.
Dashboard TIDAK melakukan pelatihan ulang, pemilihan model, atau prediksi baru.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
RESULT_PATH = ROOT / "results" / "tabel_notebook.json"
PROV_PATH = ROOT / "results" / "provenance.json"
NAVY = "#123554"
BLUE = "#3675C8"
TEAL = "#168D9E"
ORANGE = "#E99C59"
GRAY = "#60778F"

st.set_page_config(page_title="SECOM Insight | Wafer Quality Analytics", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
html, body, [data-testid="stApp"], [data-testid="stSidebar"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"]{background:linear-gradient(126deg,#FFFFFF 0%,#F7FAFF 52%,#EFF7FD 100%);}
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E1EAF4;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1400px;padding-top:2.15rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Plus Jakarta Sans',sans-serif!important;color:#123554!important;letter-spacing:-.035em;}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #E1EAF4;border-radius:16px;padding:15px 18px;box-shadow:0 3px 12px rgba(27,76,130,.035);}
[data-testid="stMetricValue"]{color:#133B64;letter-spacing:-.035em;}
[data-testid="stDataFrame"]{border:1px solid #E3EAF3;border-radius:12px;overflow:hidden;}
section[data-testid="stSidebar"] div.stButton > button {
    text-align:left!important;
    justify-content:flex-start!important;
    padding:12px 14px!important;
    font-weight:650!important;
    border-radius:11px!important;
    white-space:normal!important;
    height:auto!important;
    min-height:45px!important;
    transition: background-color .12s ease, border-color .12s ease;
}
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background:#FFFFFF!important;
    border:1px solid #E2EAF3!important;
    color:#234664!important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background:#EEF6FF!important;
    border-color:#86B5E6!important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    color:#FFFFFF!important;
    background:#286FB2!important;
    border-color:#286FB2!important;
}
[data-testid="stMainBlockContainer"] div.stButton > button {
    border-radius:10px;
}
.pill{display:inline-block;background:#ECF5FF;color:#2E69B3;font-size:.72rem;font-weight:800;border-radius:999px;padding:5px 12px;letter-spacing:.055em;}
.hero{padding:23px 27px;border-radius:19px;background:linear-gradient(108deg,#163959,#276D9F);color:white;margin-bottom:17px;}
.hero h1{color:white!important;font-size:1.9rem;margin:6px 0 9px;}
.hero p{color:#D9EAF7;margin:0;font-size:.94rem;}
.smallblock{border:1px solid #DDE8F4;background:white;border-radius:15px;padding:16px 18px;margin:8px 0 15px;line-height:1.6;}
.smallblock b{color:#143A5B;}
.sectionlabel{letter-spacing:.08em;font-weight:800;color:#47749C;font-size:.7rem;}
.stagechips{padding:9px 14px;border-radius:12px;background:#EDF4FD;border:1px solid #DDE9F7;color:#2E6399;font-weight:700;font-size:.83rem;margin:12px 0;}
.story{background:#FFFFFF;border:1px solid #DDE8F4;border-left:4px solid #3675C8;border-radius:13px;padding:19px 23px;line-height:1.78;margin:8px 0 15px;color:#204262;}
.stepnext{background:#EAF6F5;border:1px solid #D1E9E7;border-radius:13px;padding:13px 16px;line-height:1.6;color:#21545D;margin-top:15px;}
.sourcecaption{font-size:.79rem;color:#697F95;}

.prov{color:#647E97;font-size:.76rem;margin-top:-6px;margin-bottom:12px;}
div[data-testid="stAlert"]{border-radius:13px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv(DATA_PATH)
    expected = {"Time", "Pass/Fail"}
    if not expected.issubset(raw.columns):
        raise ValueError("Dataset harus memiliki kolom Time dan Pass/Fail.")
    ts = pd.to_datetime(raw["Time"], format="%Y-%m-%d %H:%M:%S")
    # Persis mengikuti prosedur koreksi tanggal pada Cell 10 notebook.
    swap = ts.dt.day <= 12
    corrected = ts.copy()
    corrected.loc[swap] = pd.to_datetime(dict(
        year=ts.loc[swap].dt.year,
        month=ts.loc[swap].dt.day,
        day=ts.loc[swap].dt.month,
        hour=ts.loc[swap].dt.hour,
        minute=ts.loc[swap].dt.minute,
        second=ts.loc[swap].dt.second,
    ))
    raw["Time"] = corrected
    labels = raw["Pass/Fail"]
    if not labels.isin([-1, 1]).all():
        raise ValueError("Label tidak sama dengan notebook (Pass=-1, Fail=1).")
    sensors = raw.drop(columns=["Time", "Pass/Fail"])
    return raw, sensors, int(swap.sum())

@st.cache_data(show_spinner=False)
def load_results():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8")), json.loads(PROV_PATH.read_text(encoding="utf-8"))

if not DATA_PATH.is_file() or not RESULT_PATH.is_file():
    st.error("File sumber tidak lengkap. Pastikan folder data/ dan results/ diunggah bersama app.py.")
    st.stop()

try:
    data, sensors, n_time_fixed = load_data()
    tables, provenance = load_results()
except Exception as exc:
    st.error(f"Tidak dapat memuat sumber: {exc}")
    st.stop()


def tab(name):
    return pd.DataFrame(tables[name])


def first_number(value):
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else np.nan


def download_table(frame, name):
    st.download_button("Unduh tabel CSV", frame.to_csv(index=False).encode("utf-8-sig"), file_name=name, mime="text/csv")


def header(kicker, title, desc):
    st.markdown(f'<div class="hero"><span style="color:#A9D5F4;font-size:.72rem;font-weight:800;letter-spacing:.14em">{kicker}</span><h1>{title}</h1><p>{desc}</p></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="smallblock">{text}</div>', unsafe_allow_html=True)


def source(cell_ix):
    st.markdown(f'<div class="prov">Sumber: notebook SECOM yang diunggah, sel indeks {cell_ix} (penomoran dari nol). Angka berasal dari output Colab yang tersimpan, bukan hasil training di dashboard.</div>', unsafe_allow_html=True)


def img(key, caption=None):
    path = ROOT / provenance.get("gambar", {}).get(key, "_unavailable")
    if path.exists():
        st.image(str(path), caption=caption or "Gambar asli hasil notebook Colab", use_container_width=True)


def model_chart(frame, metric_key, n=12):
    df = frame.copy()
    df["_nilai"] = df[metric_key].map(first_number)
    df = df.sort_values("_nilai", ascending=False).head(n)
    fig = px.bar(df.iloc[::-1], x="_nilai", y="Model/Indikator", orientation="h", text="_nilai", color_discrete_sequence=[BLUE])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(390, 34*len(df)+95), margin=dict(l=10,r=45,t=14,b=10),
                      xaxis_title=metric_key.upper(),yaxis_title="",showlegend=False)
    st.plotly_chart(fig, use_container_width=True)



from html import escape

# Semua uraian di bawah dirujukkan pada notebook Colab 84 sel, bukan hasil model yang dihitung ulang.
# Data metrik diambil dari tabel output Colab dalam results/tabel_notebook.json.
# Angka lain yang disebut dalam narasi sudah dicocokkan dengan output teks notebook sumber.

STAGES = {
 1: dict(title="Setup dan persiapan lingkungan", phase="Persiapan", purpose="Memastikan semua pustaka analitik tersedia dan proses penelitian dapat dijalankan secara konsisten.",
   method="Mempersiapkan Python dan pustaka pengolahan data, visualisasi, pemodelan, evaluasi, serta membaca berkas penelitian. Belum ada model yang dilatih pada langkah ini.",
   finding="Sel setup dalam notebook berhasil dijalankan dan perangkat analisis siap digunakan. Dataset baru dibaca pada Bagian 2.",
   story="Penelitian dimulai dengan menyiapkan lingkungan Google Colab agar seluruh analisis dapat menggunakan pustaka dan prosedur yang sama. Langkah ini belum bertujuan mencari model terbaik, melainkan memastikan proses selanjutnya dapat dijalankan secara berurutan. Setelah lingkungan siap, penelitian beralih pada pemeriksaan struktur dan pembacaan dataset SECOM yang diberikan dosen.",
   source="Sel 3–6", tables=[], figures=[]),
 2: dict(title="Memuat dataset dan memperbaiki waktu", phase="Persiapan", purpose="Mengenali data wafer serta memastikan waktu pencatatan layak digunakan untuk analisis temporal.",
   method="Memuat satu CSV berisi kolom Time, 590 fitur sensor, dan label Pass/Fail. Memeriksa urutan waktu dan membetulkan tanggal dengan komponen hari–bulan tertukar sebagaimana prosedur notebook.",
   finding="Dataset berisi 1.567 wafer, 590 sensor, 1.463 Pass dan 104 Fail. Sebanyak 604 baris waktu dikoreksi; enam lompatan waktu mundur menjadi nol. Periode pengamatan sesudah koreksi: 19 Juli–17 Oktober 2008.",
   story="Dari pemeriksaan awal diketahui bahwa dataset SECOM merupakan catatan pengukuran proses produksi wafer semikonduktor. Mayoritas wafer masuk kategori Pass dan hanya sebagian kecil yang Fail. Karena penelitian juga menguji model berdasarkan urutan waktu, kualitas informasi tanggal menjadi penting. Notebook menemukan adanya pertukaran posisi hari dan bulan pada sebagian catatan. Setelah diperbaiki, data tersusun kronologis sehingga analisis perubahan performa antarmasa dapat dilakukan dengan dasar yang lebih tepat.",
   source="Sel 7–10", tables=[], figures=[]),
 3: dict(title="Pemeriksaan kualitas data", phase="Persiapan", purpose="Mengetahui tingkat missing values, fitur konstan, dan masalah kualitas yang harus ditangani sebelum pemodelan.",
   method="Menghitung proporsi sel kosong dan jumlah fitur bermasalah. Pada tahap ini hasilnya bersifat diagnosis; keputusan pembersihan akhir baru dilakukan di dalam pipeline training pada Bagian 6.",
   finding="Sebanyak 4,5375% sel sensor kosong; 538 sensor memiliki setidaknya satu nilai kosong; 28 sensor memiliki missing lebih dari 50%; dan 116 sensor konstan pada pemeriksaan seluruh data.",
   story="Data sensor proses manufaktur tidak seluruhnya terukur pada setiap wafer. Pemeriksaan kualitas menunjukkan sebagian besar variabel mempunyai nilai hilang, tetapi proporsi sel kosong secara keseluruhan relatif kecil. Terdapat pula sensor yang hampir tidak memberikan variasi informasi. Temuan ini menjadi alasan penelitian tidak langsung memasukkan seluruh kolom ke algoritma. Namun, agar pengujian adil, aturan pembersihan tidak ditetapkan berdasarkan gabungan training dan test, melainkan dipelajari dari training pada setiap proses validasi.",
   source="Sel 11–12", tables=[("kualitas","Hasil pemeriksaan kualitas data")], figures=["missing_distribution","missing_matrix"]),
 4: dict(title="Pembagian training dan test", phase="Persiapan", purpose="Menyiapkan evaluasi yang tidak menggunakan data test untuk pemilihan model.",
   method="Menggunakan stratified split 75:25 agar prevalensi Fail relatif serupa antara training dan test. Eksplorasi inferensial, seleksi model, tuning, dan threshold dilakukan hanya dengan training.",
   finding="Training: 1.175 wafer (1.097 Pass, 78 Fail). Test acak: 392 wafer (366 Pass, 26 Fail). Proporsi Fail masing-masing sekitar 6,64% dan 6,63%.",
   story="Data penelitian dibagi menjadi dua kelompok dengan fungsi yang berbeda. Bagian training digunakan untuk mempelajari pola sensor dan memilih pendekatan model, sedangkan bagian test disimpan untuk evaluasi akhir pada data yang belum dilihat oleh proses pemilihan model. Karena jumlah wafer Fail sedikit, pembagian secara stratifikasi menjaga agar kedua kelompok tetap memiliki representasi Fail yang sebanding. Pembagian inilah yang digunakan dalam pengujian acak dan tidak boleh dicampur dengan pengujian kronologis yang dilakukan kemudian.",
   source="Sel 14–15", tables=[("split_acak","Komposisi training dan test acak")], figures=[]),
 5: dict(title="Eksplorasi data (EDA) dan pola sensor", phase="Eksplorasi", purpose="Memahami distribusi kelas, perubahan antarbulan, hubungan missingness dengan Fail, dan struktur fitur sensor.",
   method="Menganalisis distribusi kelas dan fail rate bulanan; uji Fisher dan koreksi FDR untuk missingness; Mann–Whitney U dan FDR untuk fitur tunggal; korelasi; serta PCA dan t-SNE.",
   finding="Fail rate berbeda antarbulan (uji chi-square, p < 0,001). Dari 103 fitur yang diuji untuk missingness, tidak ada yang signifikan setelah koreksi FDR. Dari 474 fitur yang diuji secara univariat, 13 signifikan setelah koreksi FDR. PCA memerlukan 126 komponen untuk menjelaskan sekitar 90% variasi dari 450 fitur yang diperiksa.",
   story="Eksplorasi memperlihatkan bahwa kesulitan utama dataset bukan sekadar banyaknya fitur, tetapi perbedaan proporsi Pass dan Fail serta pola sensor yang saling berkorelasi. Perubahan fail rate antarbulan memberikan alasan untuk melakukan pengujian kronologis pada tahap berikutnya. Sejumlah sensor menunjukkan perbedaan statistik antara Pass dan Fail secara individual, meskipun sinyal tunggal belum tentu cukup untuk membuat keputusan operasional. Tidak ditemukannya missingness yang signifikan setelah FDR pada uji awal juga tidak otomatis berarti indikator missing tidak mungkin membantu model multivariat. Karena itu, penelitian melanjutkan evaluasi dengan algoritma yang dapat menggabungkan berbagai pola sensor.",
   source="Sel 16–27", tables=[], figures=["eda_temporal","univariat","korelasi","pca_tsne"]),
 6: dict(title="Preprocessing tanpa data leakage", phase="Pemodelan", purpose="Menghasilkan masukan model yang lebih bersih tanpa membocorkan informasi dari data test.",
   method="Pipeline DataCleaner membuang fitur dengan missing di atas 50% dan fitur konstan, melakukan imputasi median, lalu menangani fitur dengan korelasi |r| > 0,95. StandardScaler, seleksi fitur, dan resampling ditempatkan di dalam fold training sesuai kebutuhan model.",
   finding="Pada contoh fit pipeline, jumlah fitur berkurang dari 590 menjadi 269 setelah pembuangan 24 fitur bermissing tinggi, 116 fitur konstan, dan 181 fitur sangat berkorelasi.",
   story="Sebelum model dilatih, data sensor perlu disederhanakan agar informasi yang digunakan benar-benar relevan dan tidak didominasi nilai hilang maupun pengulangan variabel yang sangat mirip. Notebook membangun pipeline yang mempelajari aturan pembersihan hanya dari data latih di setiap fold. Pendekatan ini penting karena preprocessing yang dilakukan sebelum pembagian data dapat membuat informasi test secara tidak langsung masuk ke proses pelatihan. Hasil pemeriksaan menunjukkan jumlah fitur dapat dipangkas secara berarti tanpa mengubah variabel target Pass dan Fail.",
   source="Sel 29–30", tables=[], figures=[]),
 7: dict(title="Metrik evaluasi dan model pembanding", phase="Pemodelan", purpose="Menetapkan ukuran keberhasilan yang relevan ketika kelas Fail sangat jarang.",
   method="Menentukan PR-AUC sebagai metrik utama, ROC-AUC sebagai pendukung, serta recall, specificity, G-mean, MCC dan accuracy. Menilai dua model dummy sebagai pembanding dan memakai Repeated Stratified 5-Fold × 2.",
   finding="Prevalensi Fail hanya 6,64%, sehingga prediksi 'selalu Pass' memiliki akurasi 93,36% tetapi recall Fail = 0. Nilai dasar PR-AUC berada di sekitar 0,066.",
   story="Pada data dengan kelas Fail sangat sedikit, akurasi tinggi bisa menyesatkan. Model yang selalu menyebut seluruh wafer sebagai Pass sudah tampak akurat karena memang hampir semua wafer lulus, padahal model tersebut sama sekali tidak mampu menangkap kegagalan. Oleh sebab itu, notebook memprioritaskan PR-AUC untuk menilai kemampuan memeringkat kelas Fail, dengan ROC-AUC sebagai pelengkap. Metrik berbasis threshold, terutama sensitivitas dan G-mean, digunakan untuk menjelaskan konsekuensi model ketika dipakai sebagai alat pendeteksi risiko.",
   source="Sel 31–32", tables=[("baseline_dummy","Perbandingan model pembanding (dummy)")], figures=[]),
 8: dict(title="Screening 19 algoritma", phase="Pemodelan", purpose="Memilih kandidat algoritma klasifikasi yang paling menjanjikan pada data training.",
   method="Membandingkan 19 keluarga algoritma dengan prosedur Cross-Validation yang seragam. Menilai PR-AUC, ROC-AUC dan metrik lain, termasuk kemampuan menangkap kelas minoritas.",
   finding="Extra Trees memperoleh PR-AUC CV sekitar 0,223 ± 0,083 dan ROC-AUC CV 0,749 ± 0,031; Easy Ensemble memperoleh PR-AUC CV sekitar 0,219. Extra Trees unggul berdasarkan PR-AUC rata-rata pada tahap screening.",
   story="Setelah preprocessing disiapkan, penelitian membandingkan berbagai algoritma dengan skema pengujian yang sama, mulai dari model linear hingga ensemble pohon keputusan. Perbandingan dilakukan melalui Cross-Validation sehingga hasil tidak bergantung hanya pada satu pembagian training. Extra Trees memperoleh PR-AUC rata-rata tertinggi di antara kandidat yang diuji dan kemudian digunakan sebagai model acuan. Namun, perbedaan skor antarmodel tidak otomatis menunjukkan keunggulan yang signifikan; hal ini diperiksa secara khusus pada Bagian 13.",
   source="Sel 34–36", tables=[("screening","Hasil screening algoritma pada training")], figures=["screening"]),
 9: dict(title="Penanganan ketidakseimbangan kelas", phase="Pemodelan", purpose="Menilai bagaimana model merespons kelas Fail yang jauh lebih sedikit dibanding Pass.",
   method="Membandingkan tanpa resampling, class weight, random under/over-sampling, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek dan SMOTE-ENN pada Logistic Regression serta Extra Trees. Resampling hanya dilakukan dalam fold training.",
   finding="Pada Extra Trees, class weight memberi PR-AUC CV 0,223 tetapi recall pada threshold default masih 0. Random under-sampling meningkatkan recall menjadi 0,648, namun PR-AUC CV turun ke 0,188 dan accuracy menjadi 0,682. Pada Logistic Regression, random under-sampling memberi recall 0,603 tetapi PR-AUC 0,159. Ini memperlihatkan trade-off, bukan satu strategi yang unggul pada seluruh metrik.",
   story="Karena Fail hanya sekitar tujuh persen, model berisiko terlalu berfokus pada kelas Pass. Penelitian membandingkan beberapa strategi penyeimbangan, baik melalui pembobotan kelas maupun teknik resampling. Tujuannya bukan sekadar membuat jumlah Pass dan Fail sama pada tampilan data, tetapi melihat apakah model menjadi lebih mampu menangkap kegagalan tanpa terlalu banyak memberikan alarm palsu. Seluruh teknik resampling dibatasi pada training agar data test tetap mencerminkan kondisi kelas yang sebenarnya.",
   source="Sel 38–39", tables=[], figures=[]),
 10: dict(title="Seleksi fitur dan reduksi dimensi", phase="Pemodelan", purpose="Mengidentifikasi cara mengurangi banyaknya sensor tanpa kehilangan informasi prediktif yang penting.",
   method="Menguji ANOVA F, mutual information, L1-logistic, Random Forest importance, dan PCA pada Logistic Regression serta Extra Trees; seluruh pemilihan fitur dijalankan dalam fold training.",
   finding="Pada Extra Trees, PR-AUC CV dengan semua fitur setelah cleaning adalah 0,223; dengan seleksi L1-logistic 0,222, ANOVA F (100 fitur) 0,211, dan PCA (95% varians) 0,137. Penyederhanaan fitur tidak otomatis meningkatkan PR-AUC.",
   story="Jumlah sensor yang besar belum tentu menjamin model semakin baik. Beberapa sensor membawa informasi yang mirip, sedangkan sensor lain memiliki hubungan lemah dengan status wafer. Karena itu, penelitian menguji seleksi fitur dan reduksi dimensi untuk melihat apakah model yang lebih sederhana tetap dapat mempertahankan kemampuan prediksinya. Interpretasi hasil tidak hanya berdasarkan berapa fitur yang dibuang, tetapi terutama apakah PR-AUC yang dihasilkan lebih baik pada validasi training.",
   source="Sel 41–42", tables=[], figures=[]),
 11: dict(title="Tuning hyperparameter", phase="Pemodelan", purpose="Menguji apakah pengaturan algoritma yang lebih tepat meningkatkan kemampuan prediksi.",
   method="Menggunakan Optuna (TPE dan median pruning) pada Extra Trees serta LightGBM, dengan PR-AUC sebagai objektif utama dalam Stratified 5-Fold Cross-Validation training.",
   finding="Dalam proses tuning, Extra Trees memperoleh PR-AUC CV terbaik 0,2110 (25 trial), sedangkan LightGBM memperoleh 0,2228 (30 trial) pada rancangan CV tuning yang digunakan.",
   story="Setelah membandingkan algoritma dasar, penelitian mencoba menyetel parameter untuk dua keluarga model yang menjanjikan. Penyetelan dilakukan menggunakan data training, bukan test, agar hasil akhir tetap dapat dievaluasi secara terpisah. Skor pada proses optimasi berguna untuk memilih konfigurasi kandidat, tetapi karena pemilihan terjadi pada data training yang sama, angka tersebut tidak dapat langsung diperlakukan sebagai bukti keunggulan final. Penilaian pada test set tetap dilakukan pada Bagian 15.",
   source="Sel 44–46", tables=[], figures=[]),
 12: dict(title="Ensemble: Soft Voting dan Stacking", phase="Pemodelan", purpose="Menilai apakah menggabungkan beberapa algoritma memberikan prediksi lebih baik dibanding model tunggal.",
   method="Menguji Soft Voting berbasis Extra Trees, LightGBM, CatBoost, dan Logistic Regression; serta Stacking dengan meta-model Logistic Regression.",
   finding="Soft Voting menghasilkan PR-AUC CV 0,208 ± 0,060 dan Stacking 0,213 ± 0,067. Extra Trees awal tetap unggul pada pembandingan PR-AUC CV yang dilaporkan (sekitar 0,223).",
   story="Ensemble mencoba memanfaatkan sudut pandang berbagai algoritma agar prediksi tidak sepenuhnya bergantung pada satu model. Dua pendekatan yang dievaluasi adalah Soft Voting, yang menggabungkan probabilitas prediksi, dan Stacking, yang menggunakan model tambahan untuk mempelajari kombinasi keluaran. Hasil menunjukkan penggabungan tidak otomatis menghasilkan skor lebih tinggi. Pada tahap ini Extra Trees awal masih memiliki PR-AUC rata-rata paling baik sehingga dipertahankan sebagai model utama.",
   source="Sel 47–48", tables=[("ensemble_top","Peringkat kandidat setelah ensemble dan tuning")], figures=[]),
 13: dict(title="Uji statistik perbandingan model", phase="Pemodelan", purpose="Membedakan kenaikan metrik yang terlihat dari keunggulan yang cukup didukung bukti statistik.",
   method="Menerapkan uji Friedman, post-hoc Nemenyi, serta corrected resampled t-test pada metrik model lintas fold yang berpasangan.",
   finding="Friedman: χ² = 100,86 dan p = 4,56 × 10⁻¹² (terdapat perbedaan di antara seluruh model). Namun selisih PR-AUC Extra Trees terhadap Easy Ensemble kecil dan tidak signifikan pada uji terkoreksi (p = 0,9448).",
   story="Peringkat model berdasarkan skor rata-rata perlu ditafsirkan bersama ketidakpastiannya. Uji Friedman menunjukkan bahwa keseluruhan algoritma tidak memiliki performa yang sama, tetapi hasil tersebut tidak berarti Extra Trees secara signifikan mengalahkan semua kandidat. Saat dibandingkan dengan pesaing terdekat melalui uji yang memperhitungkan ketergantungan antarfold, keunggulannya belum terbukti signifikan. Karena jumlah Fail terbatas, penelitian menjaga klaim pada tingkat yang didukung data.",
   source="Sel 49–50", tables=[], figures=[]),
 14: dict(title="Pemilihan model final dan threshold", phase="Evaluasi", purpose="Menentukan model acuan dan ambang keputusan yang dapat menangkap wafer Fail secara operasional.",
   method="Memilih model berdasarkan PR-AUC rata-rata training CV, lalu membandingkan threshold dari prediksi Out-of-Fold (OOF) training: G-mean, Youden, MCC, F2, target akurasi, dan threshold default 0,5.",
   finding="Extra Trees dipilih. Threshold utama berdasarkan G-mean dari OOF training sekitar 0,137 (bukan 0,5); pada OOF training sensitivitas 0,654 dan spesifisitas 0,727.",
   story="Model dengan kemampuan memberi peringkat terbaik belum tentu langsung mendeteksi wafer Fail apabila memakai threshold standar 0,5. Oleh sebab itu, penelitian memilih ambang keputusan berdasarkan probabilitas OOF pada data training. G-mean digunakan untuk menyeimbangkan sensitivitas dan spesifisitas, sedangkan threshold alternatif ditampilkan untuk menggambarkan trade-off kebutuhan produksi. Dengan cara ini, data test tidak digunakan untuk mencari threshold yang menguntungkan hasil akhir.",
   source="Sel 51–52", tables=[("threshold_train","Alternatif threshold dari training OOF")], figures=["training_threshold"]),
 15: dict(title="Evaluasi final pada test acak", phase="Evaluasi", purpose="Mengukur kinerja Extra Trees pada data test yang dipisahkan sejak awal penelitian.",
   method="Menggunakan model dan threshold yang ditentukan dari training, lalu melaporkan ROC-AUC, PR-AUC, sensitivitas, spesifisitas, confusion matrix, dan CI 95% dari bootstrap.",
   finding="Pada test acak 392 wafer dengan 26 Fail, ROC-AUC 0,798 dan PR-AUC 0,240. Dengan threshold utama sekitar 0,137: 21 Fail terdeteksi, 5 Fail terlewat, 107 alarm palsu, dan 259 Pass benar dikenali. Sensitivitas 0,808 dan spesifisitas 0,708.",
   story="Evaluasi test acak menunjukkan model Extra Trees cukup mampu membedakan wafer yang berisiko gagal dari yang lolos pada data yang tidak dipakai dalam pemilihan model. Dengan ambang yang dipilih sebelumnya, model mendeteksi 21 dari 26 kasus Fail, tetapi konsekuensinya adalah 107 wafer Pass turut ditandai sebagai Fail. Hasil tersebut memperlihatkan trade-off antara kemampuan mendeteksi risiko dan beban pemeriksaan ulang. Karena jumlah Fail pada test hanya 26, estimasi kinerja perlu dilaporkan bersama interval kepercayaan, bukan sebagai angka yang pasti berlaku pada semua periode.",
   source="Sel 53–56", tables=[("test_ci","Kinerja test dan interval kepercayaan 95%"),("test_operasi","Confusion matrix pada berbagai threshold")], figures=["roc_pr_test","test_threshold"]),
 16: dict(title="Analisis biaya kesalahan klasifikasi", phase="Evaluasi", purpose="Menghubungkan hasil prediksi dengan keputusan produksi, terutama biaya wafer Fail yang lolos pemeriksaan.",
   method="Mencoba rasio biaya false negative terhadap false positive, antara lain 5:1, 10:1, 20:1, dan 50:1. Threshold dipilih dari OOF training, lalu diperiksa pada test.",
   finding="Pada ilustrasi rasio biaya FN:FP = 10:1, biaya per wafer model pada test 0,372 dibanding 0,663 untuk strategi 'selalu Pass'. Pada 5:1, model belum lebih murah daripada strategi terbaik tanpa model.",
   story="Sebuah model klasifikasi tidak hanya dinilai dari angka statistik, melainkan juga dari dampak keputusan yang dibuat berdasarkan prediksinya. Pada produksi wafer, kegagalan yang tidak terdeteksi dapat lebih mahal daripada pemeriksaan tambahan. Notebook menguji beberapa rasio biaya untuk menggambarkan bagaimana threshold yang optimal berubah mengikuti prioritas tersebut. Hasilnya bergantung pada asumsi biaya; karena biaya nyata tidak disediakan dalam dataset, angka ini harus dibaca sebagai analisis skenario dan bukan penghematan rupiah yang telah dibuktikan di pabrik.",
   source="Sel 58–59", tables=[], figures=[]),
 17: dict(title="Validasi kronologis awal", phase="Evaluasi", purpose="Menguji ketahanan model ketika pelatihan menggunakan data lebih lama dan pengujian pada periode lebih baru.",
   method="Mengurutkan observasi berdasarkan waktu yang telah dikoreksi, melatih pada 75% wafer paling awal dan menguji pada 25% terakhir. Metrik dibandingkan dengan split acak sebagai konteks, bukan hasil pada wafer yang sama.",
   finding="Pada test kronologis awal 392 wafer (24 Fail), PR-AUC turun menjadi sekitar 0,074 dan ROC-AUC menjadi 0,570; sensitivitas sekitar 0,083. Nilainya jauh lebih rendah daripada test acak (PR-AUC 0,240; ROC-AUC 0,798).",
   story="Pengujian acak memperkirakan kinerja ketika data training dan test berasal dari campuran periode yang sama. Dalam praktik manufaktur, kondisi proses dapat berubah sepanjang waktu. Karena itu, model diuji lagi menggunakan pembagian berdasarkan tanggal: belajar dari periode awal, lalu memprediksi periode setelahnya. Hasilnya menurun cukup tajam. Temuan ini penting karena menunjukkan risiko bahwa model yang tampak baik pada pembagian acak belum tentu bertahan ketika menghadapi perubahan kondisi produksi. Hasil kronologis awal inilah yang menjadi dasar pengembangan lanjutan pada Bagian 22–28.",
   source="Sel 60–61", tables=[("split_kronologis_awal","Perbandingan test acak dan test kronologis awal")], figures=[]),
 18: dict(title="Interpretabilitas model: SHAP dan permutation", phase="Evaluasi", purpose="Mengidentifikasi sensor yang relatif berkontribusi terhadap keluaran model.",
   method="Menggunakan SHAP untuk komponen LightGBM hasil tuning dan permutation importance pada test berdasarkan PR-AUC. Membandingkan peringkat dengan hasil uji univariat.",
   finding="Sensor f59 muncul sebagai peringkat teratas pada tabel interpretasi (mean |SHAP| 0,2501; penurunan PR-AUC saat dipermutasi 0,0566). Sepuluh sensor muncul konsisten pada sedikitnya dua daftar peringkat.",
   story="Selain mengetahui besarnya performa model, penelitian juga berusaha memahami sensor mana yang paling berkaitan dengan prediksi. SHAP dan permutation importance memberi dua cara melihat kontribusi fitur, sehingga hasilnya dapat dibandingkan dengan uji statistik pada tahap EDA. Sejumlah sensor muncul konsisten dalam lebih dari satu pendekatan. Namun, karena nama sensor pada SECOM sudah dianonimkan, penelitian belum dapat menyimpulkan penyebab teknis kegagalan proses dari kode sensor tersebut tanpa informasi tambahan dari engineer produksi.",
   source="Sel 63–64", tables=[("importance","Sensor dengan kontribusi penting menurut SHAP/permutation")], figures=["shap"]),
 19: dict(title="SPC multivariat dan deteksi anomali", phase="Evaluasi", purpose="Membandingkan klasifikasi terawasi dengan pendekatan pengawasan proses yang tidak menggunakan label Fail saat pelatihan.",
   method="Mencoba Hotelling T², SPE/Q berbasis PCA, Isolation Forest, dan Local Outlier Factor sebagai novelty detection. Kondisi normal dipelajari dari wafer Pass pada training.",
   finding="Di antara metode anomali yang ditampilkan, Isolation Forest memperoleh PR-AUC sekitar 0,142 dan ROC-AUC 0,612. Extra Trees supervised sebagai pembanding tetap lebih tinggi (PR-AUC 0,240; ROC-AUC 0,798).",
   story="Pendeteksian kegagalan tidak selalu memerlukan model yang dilatih memakai label Fail. Dalam perspektif Statistical Process Control, wafer yang menyimpang dari kondisi normal dapat diberi tanda untuk pemeriksaan. Penelitian membandingkan beberapa pendekatan anomali dengan model Extra Trees yang dilatih secara terawasi. Hasil menunjukkan pendekatan anomali memberikan sinyal tambahan yang berguna sebagai perspektif pemantauan proses, tetapi pada data ini kemampuan membedakan Pass dan Fail belum menyamai model utama.",
   source="Sel 65–66", tables=[], figures=[]),
 20: dict(title="Pemeriksaan risiko data leakage", phase="Evaluasi", purpose="Menunjukkan mengapa proses preprocessing dan resampling harus dipisahkan dari data test.",
   method="Membandingkan prosedur yang sengaja keliru—SMOTE dan seleksi fitur sebelum split—dengan pipeline yang menjaga pemisahan training dan test.",
   finding="Contoh prosedur bocor menghasilkan ROC-AUC 0,998 dan PR-AUC 0,998. Prosedur benar menghasilkan ROC-AUC sekitar 0,798 dan PR-AUC 0,240. Nilai 0,998 adalah contoh yang tidak valid dan tidak boleh dilaporkan sebagai kinerja sebenarnya.",
   story="Skor model yang sangat tinggi belum tentu dapat dipercaya apabila langkah pengolahan datanya salah. Notebook sengaja mendemonstrasikan kondisi ketika seleksi fitur dan resampling dilakukan sebelum pembagian training dan test. Hasilnya hampir sempurna, tetapi itu terjadi karena informasi dari bagian yang seharusnya menjadi test telah ikut memengaruhi pelatihan. Dengan membandingkan kedua prosedur, penelitian menegaskan bahwa keabsahan proses evaluasi lebih penting daripada mengejar skor setinggi mungkin.",
   source="Sel 68–69", tables=[], figures=[]),
 21: dict(title="Kesimpulan penelitian utama", phase="Ringkasan", purpose="Merangkum temuan Bagian 1–20 sebelum memasuki eksperimen pengembangan model.",
   method="Meninjau hasil screening, validasi, threshold operasional, ketidakpastian test, interpretasi sensor, dan kelemahan generalisasi waktu. Notebook juga menyimpan model final hasil studi utama.",
   finding="Model acuan: Extra Trees. PR-AUC CV 0,223 ± 0,083; pada test acak PR-AUC 0,240 (CI95 0,134–0,394), ROC-AUC 0,798 (CI95 0,717–0,875), sensitivitas 0,808. Namun validasi kronologis awal turun menjadi ROC-AUC 0,570 dan PR-AUC 0,074.",
   story="Penelitian utama menetapkan Extra Trees sebagai kandidat terbaik berdasarkan PR-AUC rata-rata pada Cross-Validation yang dipakai. Pada test acak, model mampu mendeteksi sebagian besar wafer Fail apabila threshold diturunkan dari nilai default. Meskipun demikian, biaya alarm palsu serta rentang ketidakpastian harus dipertimbangkan. Hal yang paling penting adalah penurunan kinerja ketika model diuji pada periode yang lebih baru. Karena itu, hasil utama belum cukup untuk menyatakan bahwa model siap diterapkan secara konsisten pada proses produksi. Temuan ini menjadi titik awal eksperimen optimasi pada Bagian 22–28.",
   source="Sel 71–73", tables=[("ringkasan_awal","Ringkasan model utama sebelum optimasi")], figures=[]),
 22: dict(title="Tuning SVM RBF sebagai alternatif", phase="Pengembangan", purpose="Menguji apakah SVM RBF yang dituning dapat mengalahkan kinerja Extra Trees awal.",
   method="Memakai preprocessing yang sama, kemudian mengoptimalkan parameter SVM C dan gamma pada training dengan nested cross-validation; model terpilih dievaluasi pada test acak yang sama.",
   finding="ROC-AUC test SVM RBF tuned 0,7562 dan PR-AUC 0,1821; Extra Trees awal masing-masing 0,7982 dan 0,2396. Percobaan ini belum mengalahkan baseline.",
   story="Setelah mengetahui keterbatasan Extra Trees, penelitian mencoba SVM RBF dengan parameter yang dioptimalkan. Tujuannya bukan sekadar mengganti algoritma, tetapi memastikan SVM mendapat kesempatan yang adil melalui proses tuning pada data training. Hasil test menunjukkan performanya masih berada di bawah Extra Trees pada ROC-AUC maupun PR-AUC. Karena itu, fokus pengembangan selanjutnya beralih pada alternatif ensemble dan pemanfaatan informasi missingness.",
   source="Sel 74–76", tables=[("svm_cv","22A — Hasil Cross-Validation pada training"),("svm_test","22B — Hasil pengujian SVM pada test acak")], figures=[]),
 23: dict(title="Menguji sembilan alternatif metode", phase="Pengembangan", purpose="Mencari variasi algoritma atau kombinasi fitur yang berpotensi mengungguli Extra Trees awal.",
   method="Membandingkan sembilan kandidat melalui Cross-Validation training (23B), kemudian meninjau hasil test yang tersimpan (23C). Dua proses ini harus dibaca terpisah.",
   finding="Ensemble Extra Trees + CatBoost memiliki ROC-AUC CV 0,7498; Extra Trees + missingness 0,7483. Pada test, Ensemble Extra Trees + CatBoost memiliki ROC-AUC 0,7782, di bawah Extra Trees awal 0,7982.",
   story="Bagian 23 memperluas pencarian di luar SVM dengan membandingkan sembilan kandidat, termasuk ensemble dan fitur indikator missing. Tabel Alternatif CV menunjukkan skor rata-rata validasi di training sehingga menjadi dasar penilaian kandidat. Sesudah itu, tabel Alternatif Test menampilkan hasil pada test yang telah disimpan. Dua tabel tersebut bukan hasil yang sama dan tidak boleh dibandingkan seolah-olah berasal dari pembagian data identik. Hasil test belum menunjukkan pengganti Extra Trees yang lebih kuat, sehingga eksperimen dilanjutkan ke optimasi konfigurasi.",
   source="Sel 77", tables=[("alternatif_cv","23B — Alternatif CV: seleksi kandidat pada training"),("alternatif_test","23C — Alternatif Test: evaluasi kandidat pada test acak")], figures=[]),
 24: dict(title="Optimasi konfigurasi model lanjutan", phase="Pengembangan", purpose="Menilai variasi pengaturan Extra Trees dan Random Forest untuk memperbaiki ROC-AUC.",
   method="Menguji 20 konfigurasi kandidat pada training, lalu membandingkan konfigurasi terpilih dengan baseline pada test acak.",
   finding="Salah satu konfigurasi Extra Trees leaf3 missing 70% mencapai ROC-AUC test 0,8013 versus 0,7982 pada baseline, tetapi PR-AUC test turun menjadi 0,1942 versus 0,2396.",
   story="Bagian 24 berangkat dari hasil Bagian 23 yang belum memberikan peningkatan konsisten. Penelitian kemudian mencoba pengaturan baru pada keluarga model pohon keputusan. Sebagian konfigurasi memang menunjukkan ROC-AUC sedikit lebih tinggi, tetapi hasil tersebut disertai penurunan PR-AUC, padahal PR-AUC merupakan ukuran penting untuk kelas Fail yang jarang. Jadi, model belum dapat dikatakan lebih baik hanya karena satu metrik naik. Situasi ini mendorong percobaan penggabungan prediksi pada Bagian 25.",
   source="Sel 78", tables=[("eksperimen_24","24A — Ringkasan konfigurasi pada training"),("eksperimen_24_test","24B — Perbandingan konfigurasi pada test acak")], figures=[]),
 25: dict(title="Weighted Blending 60/40", phase="Pengembangan", purpose="Menguji apakah kombinasi probabilitas model menghasilkan keseimbangan ROC-AUC dan PR-AUC lebih baik.",
   method="Membandingkan kandidat bobot pada training dan memakai komposisi 60% Extra Trees awal + 40% indikator missing, sedangkan Extra Trees missing 70% mendapat bobot nol, pada evaluasi test acak.",
   finding="ROC-AUC test acak naik dari 0,7982 menjadi 0,8059; PR-AUC naik dari 0,2396 menjadi 0,2533. Peningkatan teramati belum berarti stabil pada setiap skenario evaluasi.",
   story="Setelah beberapa alternatif model tunggal belum menunjukkan keunggulan konsisten, penelitian mencoba menggabungkan skor prediksi model awal dengan indikator missing. Bobot dipilih menggunakan training, kemudian hasilnya dievaluasi pada test acak. Kombinasi 60/40 memberikan kenaikan kecil pada ROC-AUC dan PR-AUC dibanding Extra Trees awal. Hasil ini merupakan temuan positif dalam eksperimen acak, tetapi belum cukup untuk menyimpulkan model lebih baik secara umum. Karena itulah penelitian dilanjutkan pada pengujian ketahanan dan generalisasi pada Bagian 26 hingga 28.",
   source="Sel 79", tables=[("bobot_25","25A — Kandidat bobot pada training"),("hasil_25","25B — Hasil Weighted Blending pada test acak")], figures=[]),
 26: dict(title="Nested Cross-Validation pengembangan", phase="Validasi", purpose="Menilai kestabilan peningkatan ketika pemilihan bobot dan evaluasi dipisahkan di dalam training.",
   method="Menggunakan outer CV 5 fold × 2 pengulangan dengan inner CV 3 fold untuk memilih bobot. Test set 392 wafer tidak digunakan pada tahap ini.",
   finding="Rata-rata ROC-AUC meningkat dari 0,7495 menjadi 0,7610 (+0,0116), sedangkan PR-AUC berubah dari 0,2226 menjadi 0,2214 (−0,0012). Peningkatan tidak konsisten pada kedua metrik.",
   story="Bagian 26 mencoba menguji Weighted Blending secara lebih ketat, bukan hanya berdasarkan satu test acak. Dalam Nested Cross-Validation, pemilihan bobot dilakukan pada bagian training yang terpisah dari fold evaluasi. Hasil memperlihatkan rata-rata ROC-AUC memang naik, tetapi PR-AUC sedikit turun. Dengan demikian, kombinasi model belum terbukti memperbaiki seluruh ukuran kinerja. Temuan ini penting karena menunjukkan bahwa skor Bagian 25 tidak boleh dianggap cukup sebagai bukti keunggulan yang konsisten.",
   source="Sel 81", tables=[("ringkasan_26","26A — Perbandingan rata-rata Nested CV"),("fold_26","26B — Hasil setiap fold Nested CV")], figures=[]),
 27: dict(title="Stabilitas dan test kronologis pengembangan", phase="Validasi", purpose="Memeriksa kestabilan kenaikan antarfold dan menilai model pada periode wafer lebih baru.",
   method="Memeriksa pengaruh pengecualian fold Nested CV, memilih bobot pada validasi kronologis, lalu menguji pada test kronologis 392 wafer yang memiliki 24 Fail.",
   finding="Pada test kronologis, Extra Trees awal ROC-AUC 0,5700 dan PR-AUC 0,0737; model pengembangan ROC-AUC 0,6612 dan PR-AUC 0,1023. Bobot terpilih 0% Extra Trees awal dan 100% indikator missing, dari validasi yang hanya mempunyai empat Fail.",
   story="Pengujian kronologis meniru kondisi saat model diterapkan pada data yang datang setelah periode pelatihan. Pada data ini, indikator missing memberikan peningkatan metrik dibanding Extra Trees awal. Namun, model terpilih bukan lagi gabungan 60/40 seperti pada test acak: seluruh bobot jatuh pada indikator missing. Keputusan itu dibuat dari validasi yang hanya memiliki empat wafer Fail, sehingga ketahanannya perlu dibaca sangat hati-hati. Bagian 28 kemudian mengukur seberapa pasti peningkatan performa pada test kronologis tersebut.",
   source="Sel 82", tables=[("stabilitas_fold_27","27A — Stabilitas hasil antar-fold"),("bobot_kronologis","27B — Pemilihan bobot pada validasi kronologis"),("hasil_27","27C — Perbandingan hasil pada test kronologis")], figures=["stabilitas_27"]),
 28: dict(title="Paired Bootstrap dan kesimpulan akhir", phase="Validasi", purpose="Mengukur ketidakpastian selisih kinerja dan menyimpulkan batas bukti yang tersedia.",
   method="Melakukan paired bootstrap pada test kronologis, Leave-One-Fail-Out pada validasi empat Fail, dan 1.000 bootstrap bobot, lalu merangkum hasil test acak, Nested CV, dan test kronologis.",
   finding="Selisih ROC-AUC kronologis +0,0913 dengan CI95 −0,0224 s.d. 0,1993; selisih PR-AUC +0,0286 dengan CI95 −0,0198 s.d. 0,0783. Kedua interval melintasi nol, sehingga peningkatan belum terbukti signifikan secara statistik.",
   story="Bagian terakhir tidak bertujuan mencari model baru, melainkan memeriksa apakah kenaikan yang terlihat pada test kronologis cukup kuat untuk dipertahankan sebagai kesimpulan. Paired bootstrap menunjukkan rentang selisih ROC-AUC dan PR-AUC masih melintasi nol. Artinya, data belum cukup untuk memastikan bahwa model pengembangan selalu lebih baik. Pemilihan bobot juga stabil pada validasi yang sangat kecil, tetapi stabilitas pilihan tidak sama dengan bukti generalisasi. Kesimpulan akhir penelitian adalah bahwa indikator missing menjanjikan sebagai sumber informasi tambahan, namun keunggulannya belum signifikan dan belum konsisten di seluruh rancangan evaluasi.",
   source="Sel 83", tables=[("bootstrap_28","28A — Paired Bootstrap pada test kronologis"),("loo_28","28B — Leave-One-Fail-Out"),("bobot_bootstrap_28","28B — Stabilitas bobot 1.000 bootstrap"),("rekap_28","28C — Rekap tiga rancangan evaluasi")], figures=["paired_bootstrap_28","bobot_28"]),
}

PHASES = [
    ("Persiapan", "Bagian 1–4", "Menyiapkan data dan menjaga pemisahan training/test"),
    ("Eksplorasi", "Bagian 5", "Mengenali distribusi dan hubungan antar sensor"),
    ("Pemodelan", "Bagian 6–13", "Membandingkan dan memilih algoritma pada training"),
    ("Evaluasi", "Bagian 14–20", "Menentukan threshold, mengevaluasi, dan memeriksa risiko"),
    ("Ringkasan", "Bagian 21", "Menyimpulkan hasil penelitian utama sebelum pengembangan"),
]

TABLE_DESCRIPTIONS = {
 'kualitas': 'Kondisi missing values dan kualitas sensor sebelum preprocessing.',
 'split_acak': 'Pembagian data acak 75:25 dengan prevalensi Fail sebanding.',
 'baseline_dummy': 'Kinerja dummy untuk memperlihatkan kelemahan akurasi pada data tidak seimbang.',
 'screening': 'Hasil Cross-Validation screening algoritma, bukan hasil test.',
 'ensemble_top': 'Peringkat model dari hasil CV penelitian utama.',
 'threshold_train': 'Threshold dari prediksi Out-of-Fold training, bukan hasil tuning di test.',
 'test_ci': 'Estimasi evaluasi Extra Trees pada test acak dan interval kepercayaan.',
 'test_operasi': 'Jumlah TP, FN, FP, TN serta metrik operasional pada threshold yang telah dipilih.',
 'split_kronologis_awal': 'Perbandingan split acak dan kronologis; keduanya memiliki observasi test berbeda.',
 'importance': 'Peringkat sensor yang penting bagi model menurut beberapa metode interpretasi.',
 'ringkasan_awal': 'Hasil akhir model utama sebelum eksperimen Bagian 22.',
 'svm_cv': 'Hasil perbandingan dalam training CV; prosedur antar kandidat perlu dibedakan.',
 'svm_test': 'Evaluasi SVM hasil tuning dan Extra Trees awal pada test acak.',
 'alternatif_cv': 'Bagian 23B: Cross-Validation sembilan kandidat pada training.',
 'alternatif_test': 'Bagian 23C: evaluasi kandidat yang dibawa ke test set.',
 'eksperimen_24': 'Bagian 24: seleksi dan perbandingan konfigurasi model pada training.',
 'eksperimen_24_test': 'Bagian 24: hasil konfigurasi model pada test acak.',
 'bobot_25': 'Bagian 25: pilihan bobot dari training.',
 'hasil_25': 'Bagian 25: perbandingan Extra Trees dan Blending 60/40 di test acak.',
 'ringkasan_26': 'Bagian 26: rerata dan penyebaran hasil Nested Cross-Validation.',
 'fold_26': 'Bagian 26: kenaikan/penurunan pada setiap fold.',
 'stabilitas_fold_27': 'Bagian 27A: uji sensitivitas pada fold Nested CV.',
 'bobot_kronologis': 'Bagian 27B: kandidat bobot pada validasi kronologis, hanya empat Fail.',
 'hasil_27': 'Bagian 27C: hasil test kronologis 392 wafer, 24 Fail.',
 'bootstrap_28': 'Bagian 28A: interval kepercayaan dari selisih model yang berpasangan.',
 'loo_28': 'Bagian 28B: bobot ketika satu dari empat Fail validasi dikeluarkan.',
 'bobot_bootstrap_28': 'Bagian 28B: distribusi bobot terpilih dalam bootstrap.',
 'rekap_28': 'Bagian 28C: rekap dari tiga evaluasi berbeda, bukan penggabungan seluruh test.'
}

TABLE_TO_STAGE = {k: n for n, v in STAGES.items() for k, _ in v["tables"]}


def original_table(k, title):
    if k not in tables:
        st.warning(f"Tabel '{k}' belum tersedia pada hasil ekspor. Tidak ada angka pengganti yang dibuat.")
        return
    f = tab(k).drop(columns=["Unnamed: 0"], errors="ignore")
    st.markdown(f"**{title}**")
    st.caption(TABLE_DESCRIPTIONS.get(k, "Hasil asli yang diekspor dari notebook."))
    st.dataframe(f, use_container_width=True, hide_index=True)
    pos = provenance.get("sumber_tabel", {}).get(k)
    if pos:
        st.caption(f"Sumber Colab: Bagian {TABLE_TO_STAGE.get(k, '?')} · sel indeks {pos['cell_index']}, output {pos['output_index']} (dihitung dari nol).")
    with st.expander("Unduh data tabel ini"):
        download_table(f, f"secom_bagian_{TABLE_TO_STAGE.get(k,'x')}_{k}.csv")


def stage_figures(n, d):
    available = [k for k in d["figures"] if (ROOT / provenance.get("gambar",{}).get(k,"_not_found")).is_file()]
    if available:
        with st.expander(f"Lihat {len(available)} gambar asli dari Google Colab", expanded=(n in [5,8,15,18,28])):
            for key in available:
                img(key, "Output asli notebook Google Colab, bukan grafik rekonstruksi")


def stage_custom_charts(n):
    if n in [2,4,7]:
        a, b, c = st.columns(3)
        a.metric("Jumlah wafer", f"{len(data):,}".replace(",", "."))
        b.metric("Pass", int((data["Pass/Fail"]==-1).sum()))
        c.metric("Fail", int((data["Pass/Fail"]==1).sum()))
        chartdf = pd.DataFrame({"Kelas":["Pass","Fail"],"Jumlah":[int((data["Pass/Fail"]==-1).sum()),int((data["Pass/Fail"]==1).sum())]})
        fig=px.bar(chartdf,x="Kelas",y="Jumlah",text="Jumlah",color="Kelas",color_discrete_map={"Pass": BLUE,"Fail":ORANGE})
        fig.update_layout(height=295,showlegend=False,margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif n==3:
        x=sensors.isna().mean().sort_values(ascending=False).head(20)
        fig=px.bar(x.iloc[::-1].reset_index().rename(columns={"index":"Sensor",0:"Persentase"}),x="Persentase",y="Sensor",orientation="h",color_discrete_sequence=[BLUE])
        fig.update_layout(height=495,margin=dict(l=10,r=12,t=12,b=12),xaxis_title="Proporsi nilai hilang",yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
    elif n==5:
        df=data.assign(Bulan=data["Time"].dt.to_period("M").astype(str),Fail=(data["Pass/Fail"]==1).astype(int))
        by_month=df.groupby("Bulan",sort=True).agg(Jumlah=("Fail","size"),Fail=("Fail","sum"))
        by_month["Fail rate (%)"]=100*by_month["Fail"]/by_month["Jumlah"]
        fig=px.line(by_month.reset_index(),x="Bulan",y="Fail rate (%)",markers=True,color_discrete_sequence=[TEAL],text="Fail rate (%)")
        fig.update_traces(texttemplate="%{y:.1f}%",textposition="top center")
        fig.update_layout(height=315,yaxis_range=[0,max(26,by_month["Fail rate (%)"].max()*1.2)],margin=dict(l=10,r=25,t=28,b=15))
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("Eksplorasi sensor interaktif dari dataset asli"):
            sensor_name=st.selectbox("Pilih sensor",sensors.columns.tolist(),key="sensor_eda")
            vals=pd.to_numeric(data[sensor_name], errors="coerce")
            plotting=pd.DataFrame({"Nilai sensor":vals,"Kelas":data["Pass/Fail"].map({-1:"Pass",1:"Fail"})}).dropna()
            if not plotting.empty:
                fig=px.histogram(plotting,x="Nilai sensor",color="Kelas",barmode="overlay",nbins=45,opacity=.6,color_discrete_map={"Pass":BLUE,"Fail":ORANGE})
                fig.update_layout(height=330,margin=dict(l=12,r=12,t=20,b=12))
                st.plotly_chart(fig,use_container_width=True)
            st.caption("Visualisasi interaktif ini dihitung dari CSV asli. Tidak mengubah hasil pemodelan Colab.")
    elif n==8 and 'screening' in tables:
        model_chart(tab('screening'),'pr_auc',n=12)
    elif n==15 and 'test_operasi' in tables:
        ops=tab('test_operasi'); primary=ops.loc[ops['Model/Indikator'].astype(str).str.contains('UTAMA',case=False)]
        if not primary.empty:
            r=primary.iloc[0]; z=[[int(r['TN']),int(r['FP'])],[int(r['FN']),int(r['TP'])]]
            fig=go.Figure(go.Heatmap(z=z,x=['Prediksi Pass','Prediksi Fail'],y=['Aktual Pass','Aktual Fail'],text=z,texttemplate='%{text}',colorscale=[[0,'#E6F0FA'],[1,'#4F88CE']],showscale=False))
            fig.update_layout(height=310,margin=dict(l=20,r=20,t=20,b=20),yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig,use_container_width=True)
    elif n==17 and 'split_kronologis_awal' in tables:
        f=tab('split_kronologis_awal'); idx=f['Model/Indikator'].isin(['roc_auc','pr_auc'])
        if idx.any():
            df=f[idx].melt(id_vars='Model/Indikator',var_name='Jenis evaluasi',value_name='Nilai')
            fig=px.bar(df,x='Model/Indikator',y='Nilai',color='Jenis evaluasi',barmode='group',text='Nilai')
            fig.update_traces(texttemplate='%{text:.3f}',textposition='outside')
            fig.update_layout(height=320,yaxis_range=[0,1],margin=dict(l=20,r=20,t=24,b=20))
            st.plotly_chart(fig,use_container_width=True)
    elif n in [22,23,24,25,26,27]:
        lookup={22:('svm_test',['ROC-AUC','PR-AUC'],'Model'),23:('alternatif_cv',['ROC-AUC CV mean'],'Model'),24:('eksperimen_24_test',['ROC-AUC Test','PR-AUC Test'],'Model'),25:('hasil_25',['ROC-AUC Test','PR-AUC Test'],'Model'),26:('ringkasan_26',['ROC-AUC mean','PR-AUC mean'],'Model'),27:('hasil_27',['ROC-AUC','PR-AUC'],'Model')}
        k,metrics,idkey=lookup[n]
        if k in tables:
            frame=tab(k)
            usable=[x for x in metrics if x in frame]
            if usable and idkey in frame:
                if n==23: frame=frame.head(9)
                melted=frame.melt(id_vars=idkey,value_vars=usable,var_name='Metrik',value_name='Nilai')
                melted['Nilai']=pd.to_numeric(melted['Nilai'],errors='coerce')
                fig=px.bar(melted,x='Metrik' if len(usable)>1 else idkey,y='Nilai',color=idkey if len(usable)>1 else None,barmode='group',text='Nilai',color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_traces(texttemplate='%{text:.3f}',textposition='outside',cliponaxis=False)
                fig.update_layout(height=345 if n!=23 else 420, margin=dict(l=10,r=18,t=24,b=18),legend=dict(orientation='h',y=-.15))
                st.plotly_chart(fig,use_container_width=True)
    elif n==28 and 'bootstrap_28' in tables:
        f=tab('bootstrap_28');fig=go.Figure()
        for _,r in f.iterrows():
            low=float(r['CI 95% bawah']);up=float(r['CI 95% atas']);v=float(r['Selisih']);name=str(r['Metrik'])
            fig.add_trace(go.Scatter(x=[low,up],y=[name,name],mode='lines',line=dict(color=BLUE,width=7),showlegend=False))
            fig.add_trace(go.Scatter(x=[v],y=[name],mode='markers+text',text=[f'{v:+.4f}'],textposition='top center',marker=dict(size=13,color=TEAL),showlegend=False))
        fig.add_vline(x=0,line_color='#A85A68',line_dash='dash')
        fig.update_layout(height=290,xaxis_title='Selisih model pengembangan − Extra Trees',margin=dict(l=10,r=15,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)


def stage_screen(n, category):
    d=STAGES[n]
    header(f"GOOGLE COLAB / BAGIAN {n:02d} / {category.upper()}",d['title'],f"Tahap {n} dari 28 · {d['phase']} · {d['source']}")
    st.markdown(f'<div class="stagechips">Bagian {n} — {escape(d["phase"])} · Sumber langsung dari notebook yang sudah dieksekusi</div>',unsafe_allow_html=True)
    st.progress(n/28, text=f"Posisi pada alur Colab: Bagian {n} dari 28")
    st.markdown('### 1. Apa tujuan tahap ini?')
    st.write(d['purpose'])
    st.markdown('### 2. Apa yang dilakukan di Google Colab?')
    st.write(d['method'])
    st.markdown('### 3. Apa hasil yang diperoleh?')
    st.info(d['finding'])
    # Render structured source results, not generated experiments.
    try:
        stage_custom_charts(n)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        st.warning(f'Grafik ringkasan tahap ini tidak dapat ditampilkan ({type(exc).__name__}). Tabel dan penjelasan hasil asli tetap tersedia di bawah.')
    for k,title in d['tables']:
        original_table(k,title)
    stage_figures(n,d)
    st.markdown('### 4. Penjelasan')
    st.markdown(f'<div class="story">{escape(d["story"])}</div>',unsafe_allow_html=True)
    if n < 28:
        following=STAGES[n+1]
        st.markdown(f'<div class="stepnext"><b>Selanjutnya: Bagian {n+1} — {escape(following["title"])}</b><div>{escape(following["purpose"])}</div></div>',unsafe_allow_html=True)
    else:
        st.success('Penelitian selesai pada Bagian 28. Tidak perlu menambah eksperimen hanya untuk mengejar skor yang lebih tinggi.')
    st.caption('Angka kinerja model berasal dari hasil yang telah dieksekusi di Colab. Grafik eksploratif dataset boleh interaktif; dashboard tidak melakukan training ulang atau mengubah nilai evaluasi.')


def _select_stage(key, stage):
    # on_click runs before Streamlit builds a new frame.
    st.session_state[key] = stage


def step_picker(ids, key):
    if key not in st.session_state or st.session_state[key] not in ids:
        st.session_state[key] = ids[0]

    # Directly clickable stage navigation; the selectbox is retained as a shortcut.
    if key == 'stage_valid':
        st.markdown('**Buka bagian validasi:**')
        columns = st.columns(len(ids))
        for column, number in zip(columns, ids):
            with column:
                st.button(
                    f'Bagian {number}', key=f'{key}_direct_{number}',
                    type='primary' if st.session_state[key] == number else 'secondary',
                    on_click=_select_stage, args=(key, number), use_container_width=True,
                )

    selected=st.selectbox(
        'Pilih tahapan Google Colab',ids,key=key,
        format_func=lambda x:f'Bagian {x:02d} — {STAGES[x]["title"]}',
    )
    left, middle, right = st.columns([1, 1.6, 1])
    pos = ids.index(selected)
    with left:
        st.button(
            '← Tahap sebelumnya',key=f'{key}_previous',disabled=(pos==0),
            on_click=_select_stage,args=(key, ids[max(0,pos-1)]),
            use_container_width=True,
        )
    with middle:
        st.caption('Pilih Bagian pada tombol atau daftar, lalu baca hasil penelitian asli secara berurutan.')
    with right:
        st.button(
            'Tahap berikutnya →',key=f'{key}_next',disabled=(pos==len(ids)-1),
            on_click=_select_stage,args=(key, ids[min(len(ids)-1,pos+1)]),
            use_container_width=True,
        )
    return selected


def _select_page(page_name):
    st.session_state['secom_active_page'] = page_name


PAGES = [
    '🏠 Peta Penelitian',
    '📖 Penelitian Utama · Bagian 1–21',
    '🚀 Pengembangan Model · Bagian 22–25',
    '🛡️ Validasi Lanjutan · Bagian 26–28',
    '📚 Sumber & Jejak Colab',
]
if st.session_state.get('secom_active_page') not in PAGES:
    st.session_state['secom_active_page'] = PAGES[0]


with st.sidebar:
    st.markdown('### ◈ SECOM Insight')
    st.caption('Wafer Quality Analytics · Penelitian Google Colab')
    st.divider()
    st.caption('URUTAN PENELITIAN · KLIK MENU UNTUK MEMBUKA')
    for i, menu_name in enumerate(PAGES):
        st.button(
            menu_name, key=f'secom_sidebar_menu_{i}',
            type='primary' if st.session_state['secom_active_page']==menu_name else 'secondary',
            on_click=_select_page, args=(menu_name,), use_container_width=True,
        )
    st.divider()
    st.markdown('<span class="pill">DATA & HASIL COLAB ASLI</span>',unsafe_allow_html=True)
    st.caption('Baca penelitian utama terlebih dahulu, baru pengembangan model dan uji stabilitas.')
    st.caption('Created by Kelompok 2')

page = st.session_state['secom_active_page']

if page == '🏠 Peta Penelitian':
    header('OVERVIEW / RESEARCH JOURNEY','Alur Penelitian Kegagalan Wafer','Mulai dari notebook utama Bagian 1–21, kemudian ikuti eksperimen pengembangan dan validasi Bagian 22–28.')
    a,b,c,d=st.columns(4)
    a.metric('Jumlah wafer',f'{len(data):,}'.replace(',','.'))
    b.metric('Sensor',f'{sensors.shape[1]}')
    c.metric('Pass',f'{int((data["Pass/Fail"]==-1).sum()):,}'.replace(',','.'))
    d.metric('Fail',f'{int((data["Pass/Fail"]==1).sum()):,}'.replace(',','.'))
    st.markdown('### Urutan yang disajikan')
    for phase, number, explanation in PHASES:
        st.markdown(f'<div class="smallblock"><b>{escape(number)} · {escape(phase)}</b><div>{escape(explanation)}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="stepnext"><b>Setelah Bagian 21</b>: pengembangan model (Bagian 22–25) → validasi ketahanan dan kesimpulan (Bagian 26–28).</div>',unsafe_allow_html=True)
    st.subheader('Mengapa penelitian belum selesai pada skor test acak?')
    st.write('Extra Trees menjadi model awal berdasarkan PR-AUC Cross-Validation. Pada test acak, kinerjanya cukup menjanjikan, tetapi pada data kronologis hasilnya turun. Karena itu penelitian dilanjutkan dengan eksperimen pengembangan dan pemeriksaan ketidakpastian hasil.')
    col1,col2=st.columns(2)
    with col1:
        st.markdown('**Bagian 21 — Hasil model awal**')
        st.metric('ROC-AUC test acak','0,798')
        st.metric('PR-AUC test acak','0,240')
    with col2:
        st.markdown('**Bagian 28 — Kesimpulan pengembangan**')
        st.metric('ROC-AUC kronologis','0,6612', '+0,0913 dibanding awal')
        st.warning('CI 95% dari peningkatan masih melintasi nol. Belum signifikan.')
    st.info('Bagian 1–21 adalah penelitian utama. Bagian 22–28 adalah pengembangan dan validasi tambahan; jangan membaca angka dari rancangan test berbeda sebagai satu peringkat model.')

elif page == '📖 Penelitian Utama · Bagian 1–21':
    st.markdown('### Penelitian utama — Bagian 1 sampai 21')
    st.write('Ikuti satu tahap per halaman: tujuan, proses, hasil, lalu narasi penjelasannya. Data dan skor evaluasi sesuai notebook Colab.')
    chosen=step_picker(list(range(1,22)),'stage_main')
    stage_screen(chosen,'Penelitian utama')
    if chosen==21:
        st.divider()
        st.success('Penelitian utama selesai. Silakan lanjutkan ke menu Pengembangan Model (Bagian 22–25).')

elif page == '🚀 Pengembangan Model · Bagian 22–25':
    st.markdown('### Pengembangan setelah model awal ditetapkan')
    st.write('Alurnya: tuning SVM → sembilan alternatif metode → optimasi konfigurasi → Weighted Blending. Semua tabel diberi nomor Bagian dan fungsi evaluasinya.')
    chosen=step_picker([22,23,24,25],'stage_dev')
    stage_screen(chosen,'Pengembangan model')
    if chosen==25:
        st.success('Setelah hasil blending test acak, lanjutkan ke Validasi Lanjutan (Bagian 26–28).')

elif page == '🛡️ Validasi Lanjutan · Bagian 26–28':
    st.markdown('### Menguji seberapa kuat peningkatan yang ditemukan')
    st.write('Urutannya: Nested CV pada training → test kronologis → paired bootstrap dan stabilitas bobot. Test acak dan kronologis dijelaskan sebagai dua konteks berbeda.')
    chosen=step_picker([26,27,28],'stage_valid')
    stage_screen(chosen,'Validasi lanjutan')

elif page == '📚 Sumber & Jejak Colab':
    header('AUDIT / TRACEABILITY','Jejak Google Colab','Lihat tabel dan gambar beserta nomor Bagian asal, bukan hanya nama teknis ekspor.')
    st.markdown('### Sumber penelitian')
    st.caption('Pilih tabel atau gambar pada bagian di bawah untuk memeriksa asal hasilnya. Anda juga dapat mengunduh berkas sumber yang tersedia.')
    source_files = [
        ('Dataset penelitian (CSV)', DATA_PATH, 'text/csv'),
        ('Tabel hasil asli Colab (JSON)', RESULT_PATH, 'application/json'),
        ('Pemetaan jejak sel Colab (JSON)', PROV_PATH, 'application/json'),
        ('Notebook Colab (IPYNB)', ROOT / 'notebook' / 'SECOM_Colab_Lengkap_SVM.ipynb', 'application/json'),
    ]
    with st.expander('Buka berkas sumber dan unduh', expanded=False):
        for title, path, mime in source_files:
            if path.is_file():
                st.download_button(
                    f'Unduh {title}', data=path.read_bytes(), file_name=path.name,
                    mime=mime, key=f'secom_source_{path.name}', use_container_width=True,
                )
            else:
                st.caption(f'{title}: berkas belum tersedia di repository.')
    st.write('**Dataset:** `data/data.csv` — sumber EDA dengan 1.567 wafer dan 590 variabel sensor.')
    st.write('**Hasil:** `results/tabel_notebook.json` — tabel output yang telah dijalankan dan diekspor dari notebook.')
    st.write('**Gambar:** `results/figures/` — visualisasi asli yang tersimpan dari output notebook.')
    st.write('**Notebook:** `notebook/SECOM_Colab_Lengkap_SVM.ipynb` — arsip sel penelitian ketika folder notebook juga tersedia di GitHub.')
    st.info('Tidak ada pelatihan ulang, pemilihan bobot ulang, atau prediksi wafer baru di dashboard. Jika hasil Colab diubah, output yang diekspor harus diperbarui.')
    choices=sorted(tables.keys(),key=lambda x:(TABLE_TO_STAGE.get(x,99),x))
    if choices:
        k=st.selectbox('Pilih hasil berdasarkan Bagian Colab',choices,format_func=lambda x:f'Bagian {TABLE_TO_STAGE.get(x,"?"):>2} — {TABLE_DESCRIPTIONS.get(x,x)}')
        original_table(k,f'Output asli: Bagian {TABLE_TO_STAGE.get(k,"?")}')
    st.markdown('### Gambar asli')
    pics=sorted(provenance.get('gambar',{}).keys())
    if pics:
        k=st.selectbox('Pilih gambar output',pics)
        img(k,f'Gambar asli notebook: {k}')
    st.markdown('### Cara membaca sumber evaluasi')
    st.markdown('**CV** = pengujian berulang pada training. **Test acak** = 392 wafer dengan 26 Fail, digunakan untuk evaluasi model awal dan sebagian eksperimen. **Test kronologis** = 392 wafer dari periode akhir dengan 24 Fail. Angkanya tidak boleh disamakan, karena observasi dan prosesnya berbeda.')
