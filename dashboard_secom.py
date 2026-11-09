"""SECOM Insight | Dashboard hasil eksperimen Google Colab (Bagian 1–28).

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
section[data-testid="stSidebar"] .stRadio label{margin-bottom:.28rem;}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{padding:9px 12px!important;border-radius:10px;}
.pill{display:inline-block;background:#ECF5FF;color:#2E69B3;font-size:.72rem;font-weight:800;border-radius:999px;padding:5px 12px;letter-spacing:.055em;}
.hero{padding:23px 27px;border-radius:19px;background:linear-gradient(108deg,#163959,#276D9F);color:white;margin-bottom:17px;}
.hero h1{color:white!important;font-size:1.9rem;margin:6px 0 9px;}
.hero p{color:#D9EAF7;margin:0;font-size:.94rem;}
.smallblock{border:1px solid #DDE8F4;background:white;border-radius:15px;padding:16px 18px;margin:8px 0 15px;line-height:1.6;}
.smallblock b{color:#143A5B;}
.sectionlabel{letter-spacing:.08em;font-weight:800;color:#47749C;font-size:.7rem;}
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


with st.sidebar:
    st.markdown("### ◈ SECOM Insight")
    st.caption("Wafer Quality Analytics · Penelitian Google Colab")
    st.divider()
    page = st.radio("NAVIGASI PENELITIAN", [
        "🏠 Ringkasan", "📂 Dataset & EDA", "📊 Benchmark Model",
        "🎯 Evaluasi Test", "🧪 Pengembangan 22–25", "🛡️ Validasi 26–28", "📚 Jejak Colab",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<span class="pill">HASIL EMPIRIS · BAGIAN 1–28</span>', unsafe_allow_html=True)
    st.caption("Dataset otomatis dari data/data.csv. Hasil model dibaca dari output notebook. Tidak ada training ulang atau data simulasi.")
    st.caption("Created by Kelompok 2")

if page == "🏠 Ringkasan":
    header("OVERVIEW / SECOM", "Klasifikasi Kegagalan Wafer", "Mengkomunikasikan hasil eksperimen SECOM berdasarkan dataset dosen dan notebook yang telah dijalankan.")
    n_fail = int((data["Pass/Fail"] == 1).sum())
    a,b,c,d=st.columns(4)
    a.metric("Jumlah wafer", f"{len(data):,}".replace(",","."))
    b.metric("Sensor", f"{sensors.shape[1]}")
    c.metric("Wafer Fail", f"{n_fail}", f"{n_fail/len(data)*100:.2f}% dari total", delta_color="off")
    d.metric("Sel sensor missing", f"{sensors.isna().to_numpy().mean()*100:.2f}%")
    left,right=st.columns([1,1.4], gap="large")
    with left:
        st.subheader("Distribusi kelas")
        counts=pd.DataFrame({"Kelas":["Pass (-1)","Fail (1)"], "Jumlah":[int((data["Pass/Fail"]==-1).sum()),n_fail]})
        fig=px.pie(counts, names="Kelas", values="Jumlah", hole=.63, color="Kelas",color_discrete_map={"Pass (-1)":BLUE,"Fail (1)":ORANGE})
        fig.update_layout(height=310, margin=dict(l=10,r=10,t=10,b=10),legend=dict(orientation="h",y=-.05))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Tiga konteks pengujian")
        rekap=tab("rekap_28")
        st.dataframe(rekap[["Tahap","ROC-AUC awal","ROC-AUC baru","PR-AUC awal","PR-AUC baru"]],hide_index=True,use_container_width=True)
        note("<b>Temuan utama.</b> Model pengembangan naik pada test acak dan test kronologis, tetapi peningkatan kronologis <b>belum signifikan pada paired bootstrap 95%</b>. Nested CV memberi hasil campuran: ROC-AUC naik, PR-AUC sedikit turun.")
    st.subheader("Alur penelitian yang sudah diselesaikan")
    st.markdown("**Persiapan & EDA** → **Screening 19 algoritma** → **Penyetelan model & evaluasi** → **Pengembangan (Bagian 22–25)** → **Nested CV & kronologis (26–27)** → **Bootstrap & stabilitas (28)**")
    st.info("Dashboard menyajikan hasil penelitian yang sudah selesai. Perubahan filter EDA tidak mengubah metrik model dari eksperimen Colab.")

elif page == "📂 Dataset & EDA":
    header("DATASET / EKSPLORASI", "Memahami Data SECOM", "Visualisasi dihitung langsung dari satu CSV dosen, dengan koreksi kolom Time mengikuti notebook Colab.")
    a,b,c,d=st.columns(4)
    a.metric("Observasi",len(data)); b.metric("Fitur sensor",len(sensors.columns))
    c.metric("Fitur memiliki missing",int((sensors.isna().sum()>0).sum()))
    d.metric("Tanggal terkoreksi",n_time_fixed)
    st.markdown('<div class="prov">Koreksi tanggal mengikuti cell 10 notebook. Bukan mengubah nilai sensor atau label Pass/Fail.</div>',unsafe_allow_html=True)
    dmin,dmax=data["Time"].min().date(),data["Time"].max().date()
    selection=st.date_input("Rentang waktu untuk eksplorasi", value=(dmin,dmax),min_value=dmin,max_value=dmax)
    if not isinstance(selection,(tuple,list)) or len(selection)!=2:
        st.info("Pilih tanggal awal dan akhir untuk memfilter grafik EDA.")
        view=data
    else:
        view=data[(data["Time"].dt.date>=selection[0]) & (data["Time"].dt.date<=selection[1])]
    if len(view)==0:
        st.warning("Tidak ada wafer pada rentang waktu terpilih.")
        st.stop()
    ll,rr=st.columns([1,1.1],gap="large")
    with ll:
        st.subheader("Pass/Fail pada periode terpilih")
        grp=view["Pass/Fail"].map({-1:"Pass",1:"Fail"}).value_counts().reindex(["Pass","Fail"],fill_value=0)
        fig=px.bar(x=grp.index,y=grp.values,labels={"x":"Kelas wafer","y":"Jumlah wafer"},color=grp.index,color_discrete_map={"Pass":BLUE,"Fail":ORANGE},text=grp.values)
        fig.update_layout(showlegend=False,height=330,margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with rr:
        st.subheader("Proporsi Fail menurut bulan")
        monthly=view.assign(Bulan=view["Time"].dt.to_period("M").astype(str)).groupby("Bulan")["Pass/Fail"].agg(
            Jumlah="size",Fail=lambda x: int((x==1).sum()))
        monthly["Fail_rate"]=100*monthly["Fail"]/monthly["Jumlah"]
        fig=px.line(monthly.reset_index(),x="Bulan",y="Fail_rate",markers=True,text="Fail_rate",
                    hover_data=["Jumlah","Fail"],labels={"Fail_rate":"Persentase Fail (%)"})
        fig.update_traces(line_color=TEAL,marker_size=10,texttemplate="%{y:.1f}%",textposition="top center")
        fig.update_layout(height=330,margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Kualitas data sensor")
    missing=(view.drop(columns=["Time","Pass/Fail"]).isna().mean()*100).sort_values(ascending=False)
    a,b=st.columns([1.1,1],gap="large")
    with a:
        top=missing.head(20).reset_index();top.columns=["Sensor","Missing (%)"]
        fig=px.bar(top.iloc[::-1],x="Missing (%)",y="Sensor",orientation="h",color_discrete_sequence=[BLUE])
        fig.update_layout(height=540,margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        selected=st.selectbox("Pilih sensor untuk melihat distribusi",options=list(sensors.columns),index=0)
        subset=view[[selected,"Pass/Fail"]].dropna().copy()
        subset["Kelas"]=subset["Pass/Fail"].map({-1:"Pass",1:"Fail"})
        if not subset.empty and subset[selected].nunique()>1:
            fig=px.histogram(subset,x=selected,color="Kelas",barmode="overlay",nbins=35,
                             opacity=.65,color_discrete_map={"Pass":BLUE,"Fail":ORANGE},
                             labels={selected:f"Nilai sensor {selected}"})
            fig.update_layout(height=350,margin=dict(l=10,r=10,t=15,b=10))
            st.plotly_chart(fig,use_container_width=True)
        else:st.info("Sensor ini konstan atau tidak memiliki data terukur pada rentang yang dipilih.")
        st.caption("Nama sensor mengikuti file dosen (0–589); tidak ada informasi proses fisik untuk menafsirkan fungsi tiap sensor.")
    with st.expander("Lihat data asli dan ringkasan kualitas dari Colab"):
        st.dataframe(view.head(25),use_container_width=True)
        st.dataframe(tab("kualitas"),hide_index=True,use_container_width=True)
        source(12)
    with st.expander("Lihat gambar original EDA dari notebook"):
        img("eda_temporal");img("missing_distribution");img("missing_matrix")

elif page == "📊 Benchmark Model":
    header("MODEL COMPARISON", "Screening Algoritma Klasifikasi", "Metrik berasal dari cross-validation training; model dummy ditampilkan sebagai pembanding, bukan algoritma utama.")
    screen=tab("screening")
    screen=screen[~screen["Model/Indikator"].astype(str).str.startswith("Dummy")].copy()
    left,right=st.columns([2,1])
    with left: metric=st.selectbox("Urutkan berdasarkan metrik",["pr_auc","roc_auc","recall","g_mean","accuracy","mcc"])
    with right: n=st.slider("Jumlah model ditampilkan",5,len(screen),min(12,len(screen)))
    model_chart(screen,metric,n)
    note("<b>Cara membaca.</b> Dataset sangat tidak seimbang (hanya 6,64% Fail). PR-AUC lebih informatif dibanding akurasi untuk menilai kemampuan memeringkat wafer Fail. ROC-AUC tetap disajikan sebagai metrik pembanding.")
    st.subheader("Tabel lengkap screening")
    st.dataframe(screen,hide_index=True,use_container_width=True)
    download_table(screen,"screening_model_colab.csv")
    source(35)
    with st.expander("Gambar benchmark dari notebook"):
        img("screening")

elif page == "🎯 Evaluasi Test":
    header("FINAL EVALUATION / INITIAL MODEL", "Evaluasi Extra Trees Awal", "Evaluasi test acak stratified, menggunakan model dan ambang keputusan yang dipilih dari data training di notebook.")
    ci=tab("test_ci")
    def get_ci(metric):
        r=ci[ci["Model/Indikator"].str.lower()==metric].iloc[0]
        return float(r["estimasi"]),float(r["CI95_bawah"]),float(r["CI95_atas"])
    ra=get_ci("roc_auc");pa=get_ci("pr_auc")
    a,b,c=st.columns(3)
    a.metric("ROC-AUC test",f"{ra[0]:.3f}")
    b.metric("PR-AUC test",f"{pa[0]:.3f}")
    c.metric("Jumlah wafer test acak",392)
    st.caption(f"ROC-AUC CI95: {ra[1]:.3f}–{ra[2]:.3f}; PR-AUC CI95: {pa[1]:.3f}–{pa[2]:.3f}. Keduanya mengikuti tabel bootstrap notebook Bagian 15.")
    st.subheader("Confusion matrix berdasarkan titik operasi hasil Colab")
    ops=tab("test_operasi")
    selected=st.selectbox("Ambang keputusan",ops["Model/Indikator"].tolist())
    rr=ops[ops["Model/Indikator"]==selected].iloc[0]
    l,r=st.columns([1.2,1],gap="large")
    with l:
        mat=[[int(rr["TN"]),int(rr["FP"])],[int(rr["FN"]),int(rr["TP"])]]
        heat=go.Figure(go.Heatmap(z=mat,x=["Prediksi Pass","Prediksi Fail"],y=["Aktual Pass","Aktual Fail"],
                                  text=mat,texttemplate="%{text}",textfont_size=19,colorscale=[[0,"#F5FAFF"],[1,"#3478C5"]],showscale=False))
        heat.update_layout(height=350,margin=dict(l=15,r=15,t=15,b=15),yaxis=dict(autorange="reversed"))
        st.plotly_chart(heat,use_container_width=True)
    with r:
        aa,bb=st.columns(2)
        aa.metric("Recall Fail",f'{float(rr["sensitivity"])*100:.1f}%')
        bb.metric("Precision",f'{float(rr["precision"])*100:.1f}%')
        aa.metric("False positive",int(rr["FP"]))
        bb.metric("Fail terdeteksi",int(rr["TP"]))
        note("Makin rendah threshold, jumlah Fail terdeteksi dapat meningkat, tetapi false alarm biasanya ikut meningkat. <b>Tabel ini untuk Extra Trees awal pada test acak</b>, bukan untuk model kronologis Bagian 27.")
    st.dataframe(ops,hide_index=True,use_container_width=True)
    download_table(ops,"test_extra_trees_operating_points.csv")
    source(55)
    with st.expander("Kurva ROC / PR dan visual original notebook"):
        img("roc_pr_test")
        img("test_threshold")

elif page == "🧪 Pengembangan 22–25":
    header("EXPERIMENTAL DEVELOPMENT", "Eksperimen Pengembangan Model", "Perbandingan model tambahan dan Weighted Blending menggunakan hasil notebook, tanpa melatih ulang model.")
    t1,t2,t3=st.tabs(["SVM RBF tuned (22)","Alternatif & optimasi (23–24)","Weighted Blending (25)"])
    with t1:
        st.subheader("SVM RBF dan baseline Extra Trees")
        svm=tab("svm_test")
        st.dataframe(svm.drop(columns=["Unnamed: 0"],errors="ignore"),hide_index=True,use_container_width=True)
        fig=px.bar(svm.melt(id_vars="Model",value_vars=["ROC-AUC","PR-AUC"],var_name="Metrik",value_name="Nilai"),
                   x="Metrik",y="Nilai",color="Model",barmode="group")
        fig.update_layout(height=380,margin=dict(l=10,r=10,t=18,b=10))
        st.plotly_chart(fig,use_container_width=True)
        note("Hasil test menunjukkan SVM RBF tuned belum melampaui baseline Extra Trees. Data test ini adalah bagian evaluasi eksperimen lanjutan, bukan alasan untuk memilih model kembali berdasarkan hasil test.")
        source(75)
    with t2:
        st.subheader("Sembilan alternatif pada training")
        alt=tab("alternatif_cv")
        st.dataframe(alt,hide_index=True,use_container_width=True)
        st.subheader("Ringkasan optimasi Bagian 24")
        st.dataframe(tab("eksperimen_24"),hide_index=True,use_container_width=True)
        st.dataframe(tab("eksperimen_24_test"),hide_index=True,use_container_width=True)
        source(78)
    with t3:
        hasil=tab("hasil_25")
        left,right=st.columns(2)
        with left:
            st.subheader("Evaluasi hasil blending")
            st.dataframe(hasil,hide_index=True,use_container_width=True)
        with right:
            st.subheader("Bobot terpilih")
            st.metric("Extra Trees awal","60%")
            st.metric("Indikator missing","40%")
            st.caption("Komponen ET missing 70% memperoleh bobot 0%. Bobot dipilih pada training sebagaimana output Bagian 25.")
        melt=hasil.melt(id_vars="Model",value_vars=["ROC-AUC Test","PR-AUC Test"],var_name="Metrik",value_name="Nilai")
        fig=px.bar(melt,x="Metrik",y="Nilai",color="Model",barmode="group",text="Nilai")
        fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
        fig.update_layout(height=390,margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig,use_container_width=True)
        note("Blending 60/40 meningkatkan metrik test acak dibanding Extra Trees awal; Bagian 26–28 memeriksa konsistensi temuan ini menggunakan rancangan evaluasi yang berbeda.")
        with st.expander("Lihat kandidat bobot yang diuji"):
            st.dataframe(tab("bobot_25"),hide_index=True,use_container_width=True)
        source(79)

elif page == "🛡️ Validasi 26–28":
    header("ROBUSTNESS CHECKS", "Validasi dan Stabilitas Model", "Membedakan Nested CV, test kronologis, serta ketidakpastian peningkatan model pada paired bootstrap.")
    t1,t2,t3=st.tabs(["Nested CV · 26","Kronologis · 27","Bootstrap · 28"])
    with t1:
        st.subheader("Nested Cross-Validation 5×2 pada training")
        cv=tab("ringkasan_26")
        st.dataframe(cv,hide_index=True,use_container_width=True)
        fold=tab("fold_26")
        fig=px.bar(fold,x="Fold",y="Selisih ROC",color="Selisih ROC",color_continuous_scale="RdBu")
        fig.add_hline(y=0,line_dash="dash",line_color=GRAY)
        fig.update_layout(height=350,margin=dict(l=10,r=10,t=10,b=10),coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Kenaikan ROC-AUC mean dari 0,7495 menjadi 0,7610; PR-AUC mean turun dari 0,2226 menjadi 0,2214. Bukan perbandingan langsung dengan test kronologis.")
        with st.expander("Per fold dan uji sensitivitas"):
            st.dataframe(fold,hide_index=True,use_container_width=True)
            st.dataframe(tab("stabilitas_fold_27"),hide_index=True,use_container_width=True)
            img("stabilitas_27")
        source(81)
    with t2:
        st.subheader("Pengujian kronologis pada periode akhir")
        a,b,c=st.columns(3)
        a.metric("Training kronologis","940", "Fail 76",delta_color="off")
        b.metric("Validasi kronologis","235", "Fail 4",delta_color="off")
        c.metric("Test kronologis","392", "Fail 24",delta_color="off")
        cron=tab("hasil_27")
        st.dataframe(cron,hide_index=True,use_container_width=True)
        fig=px.bar(cron.melt(id_vars="Model",value_vars=["ROC-AUC","PR-AUC"],var_name="Metrik",value_name="Nilai"),
                   x="Metrik",y="Nilai",color="Model",barmode="group",text="Nilai")
        fig.update_traces(texttemplate="%{text:.4f}",textposition="outside")
        fig.update_layout(height=360,margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
        note("<b>Penting:</b> Model pengembangan pada test kronologis menggunakan <b>100% indikator missing</b> dan <b>0% Extra Trees awal</b>. Validasi bobot hanya memuat 4 Fail, sehingga klaim generalisasi perlu dibatasi.")
        with st.expander("Perbandingan bobot pada validasi kronologis"):
            st.dataframe(tab("bobot_kronologis"),hide_index=True,use_container_width=True)
        source(82)
    with t3:
        st.subheader("Paired bootstrap — ketidakpastian selisih performa")
        boot=tab("bootstrap_28")
        st.dataframe(boot,hide_index=True,use_container_width=True)
        fig=go.Figure()
        for _,rr in boot.iterrows():
            delta=float(rr["Selisih"]); low=float(rr["CI 95% bawah"]);high=float(rr["CI 95% atas"])
            fig.add_trace(go.Scatter(x=[low,high],y=[rr["Metrik"]]*2,mode="lines",line=dict(color=BLUE,width=8),showlegend=False,
                                     hovertemplate="CI 95%: %{x:.4f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[delta],y=[rr["Metrik"]],mode="markers+text",text=[f"{delta:+.4f}"],textposition="top center",
                                     marker=dict(color=TEAL,size=13),showlegend=False,hovertemplate="Selisih: %{x:.4f}<extra></extra>"))
        fig.add_vline(x=0,line_dash="dash",line_color="#B64A55")
        fig.update_layout(height=310,margin=dict(l=10,r=25,t=35,b=20),xaxis_title="Selisih model pengembangan − Extra Trees awal",yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
        st.warning("Interval kepercayaan ROC-AUC dan PR-AUC sama-sama melintasi nol. Peningkatan belum signifikan secara statistik; bootstrap ini eksploratif dan tidak mengoreksi pemilihan model sebelumnya.")
        l,r=st.columns(2)
        with l:
            st.subheader("Leave-One-Fail-Out")
            st.dataframe(tab("loo_28"),hide_index=True,use_container_width=True)
        with r:
            st.subheader("Distribusi bobot bootstrap")
            st.dataframe(tab("bobot_bootstrap_28"),hide_index=True,use_container_width=True)
            st.metric("Median bobot ET awal", "0,0")
            st.caption("1000 bootstrap bobot: semua memilih bobot 0 untuk Extra Trees awal.")
        with st.expander("Gambar asli Colab Bagian 28"):
            img("paired_bootstrap_28");img("bobot_28")
        source(83)
    st.divider()
    st.subheader("Perbandingan tiga tahap — jangan digabung sebagai satu skor")
    recap=tab("rekap_28")
    st.dataframe(recap,hide_index=True,use_container_width=True)
    st.caption("Test acak, Nested CV, dan test kronologis memakai prosedur evaluasi berbeda. Metrik ditampilkan berdampingan untuk konteks, bukan pemeringkatan langsung lintas split.")

else:
    header("AUDIT / NOTEBOOK", "Jejak Penelitian dan Reproduksibilitas", "Setiap tabel evaluasi dashboard dapat ditelusuri ke output sel notebook asal.")
    st.subheader("Sumber yang digunakan")
    st.markdown("- **data/data.csv** — dataset dosen yang sama dengan Colab: 1.567 × 592 kolom.\n- **notebook/SECOM_Colab_Lengkap_SVM.ipynb** — notebook lengkap dengan output eksekusi sampai Bagian 28.\n- **results/tabel_notebook.json** — ekspor hasil tabel dari output notebook, tanpa menghitung ulang model.\n- **results/figures/** — gambar yang diekspor dari output Colab.\n- **tools/ekspor_hasil_colab.py** — meregenerasi berkas hasil dari notebook yang sudah dieksekusi.")
    st.info("Tidak ada file model .joblib yang disertakan. Karena itu dashboard tidak melakukan prediksi wafer baru, evaluasi model baru, atau retraining. Grafik evaluasi test bersumber dari output notebook yang tersimpan.")
    st.subheader("Telusuri tabel sumber")
    avail=sorted(provenance["sumber_tabel"].keys())
    name=st.selectbox("Pilih tabel output Colab",avail,index=avail.index("rekap_28"))
    pos=provenance["sumber_tabel"][name]
    st.caption(f"Sumber: cell index {pos['cell_index']}, output index {pos['output_index']} pada file .ipynb")
    frame=tab(name)
    st.dataframe(frame,hide_index=True,use_container_width=True)
    download_table(frame,f"secom_{name}.csv")
    st.subheader("Lihat gambar output penelitian")
    pic=st.selectbox("Pilih gambar",sorted(provenance["gambar"].keys()),index=0)
    img(pic,provenance["gambar"][pic])
    st.markdown("---")
    st.caption("Cara kerja: saat situs dimuat, dataset dibaca otomatis. Output penelitian tidak dilatih ulang. Eksperimen replikasi atau analisis dataset berbeda harus dilakukan di Colab dan hasilnya diekspor kembali.")
