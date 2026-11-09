"""SECOM Insight — dashboard interaktif hasil notebook klasifikasi wafer.

Jalankan: streamlit run dashboard_secom.py
Sumber metrik penelitian: notebook SECOM yang disampaikan pengguna (Nov 2026).
Angka benchmark statis berasal dari notebook. Unggah CSV hanya untuk EDA interaktif;
metrik benchmark TIDAK dihitung ulang dari CSV yang diunggah.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SECOM Insight | Wafer Quality Analytics",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Rekap hasil yang sudah dihitung di notebook sumber, bukan hasil training baru.
# -----------------------------------------------------------------------------
STUDY = {
    "wafer": 1567, "features": 590, "pass": 1463, "fail": 104,
    "train": 1175, "test": 392, "test_fail": 26, "test_pass": 366,
    "pr_auc_cv": 0.223, "pr_auc_cv_sd": 0.083,
    "pr_auc_test": 0.240, "pr_auc_ci": (0.134, 0.394),
    "roc_auc_test": 0.798, "roc_auc_ci": (0.717, 0.875),
    "threshold": 0.1372,
}

# Semua 19 model pada screening 5-fold x 2 repeat di notebook.
# (nama, PR-AUC, simpangan baku PR-AUC, ROC-AUC, G-mean pada threshold default,
#  akurasi pada threshold default)
MODEL_RESULTS = [
    ("Extra Trees", .223, .083, .749, .000, .934),
    ("Easy Ensemble", .219, .079, .712, .632, .652),
    ("Balanced Random Forest", .217, .071, .735, .469, .879),
    ("Random Forest", .211, .066, .733, .000, .934),
    ("XGBoost", .194, .061, .702, .107, .932),
    ("Logistic Regression (L1)", .189, .063, .689, .611, .771),
    ("LightGBM", .186, .050, .703, .025, .934),
    ("MLP + SMOTE", .185, .072, .670, .399, .893),
    ("CatBoost", .180, .041, .690, .156, .928),
    ("LDA (shrinkage)", .179, .061, .679, .552, .810),
    ("Logistic Regression (L2)", .179, .060, .668, .539, .812),
    ("Linear SVM", .178, .059, .667, .600, .722),
    ("SVM RBF", .172, .042, .667, .172, .930),
    ("HistGradientBoosting", .171, .041, .659, .050, .933),
    ("RUSBoost", .118, .034, .635, .426, .801),
    ("KNN (k=15) + SMOTE", .104, .026, .621, .324, .168),
    ("Decision Tree", .102, .048, .570, .502, .778),
    ("AdaBoost (stump berbobot)", .098, .015, .632, .501, .754),
    ("Gaussian Naive Bayes", .095, .020, .567, .513, .440),
]

# Evaluasi test set Extra Trees; threshold ditetapkan dari OOF train.
# (nama, threshold, TP, FN, FP, TN, sensitivity, specificity, precision, G-mean,
#  MCC, accuracy)
OPERATING_POINTS = [
    ("G-mean maksimum (utama)", .1372, 21, 5, 107, 259, .808, .708, .164, .756, .273, .714),
    ("F2 maksimum", .150, 19, 7, 76, 290, .731, .792, .200, .761, .304, .788),
    ("MCC maksimum", .193, 6, 20, 31, 335, .231, .915, .162, .460, .124, .870),
    ("Akurasi ≥ 90%", .217, 5, 21, 17, 349, .192, .954, .227, .428, .158, .903),
    ("Default 0,50", .500, 0, 26, 0, 366, .000, 1.000, .000, .000, .000, .934),
]

TOP_SENSORS = [
    ("f103", .6779, .0001), ("f59", .6694, .0001),
    ("f247", .6699, .0050), ("f510", .6375, .0050),
    ("f519", .6687, .0050), ("f477", .6354, .0050),
    ("f205", .6332, .0056), ("f341", .6236, .0155),
    ("f95", .6112, .0233), ("f21", .6179, .0233),
    ("f33", .6167, .0243), ("f385", .6395, .0330),
]
IMPORTANT_SENSORS = ["f59", "f33", "f103", "f205", "f21", "f247", "f130", "f213", "f587", "f129"]
MONTHLY_FAIL = [
    ("Jul 2008", .2222), ("Agu 2008", .0919),
    ("Sep 2008", .0288), ("Okt 2008", .0613),
]

BG = "#F4F7FB"
NAVY = "#102A4C"
BLUE = "#2B63DA"
CYAN = "#11A7B7"
CORAL = "#EA6874"
MUTED = "#71839A"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {
  font-family:'DM Sans',sans-serif; color:#18324D !important; }
[data-testid="stAppViewContainer"] { background:#F4F7FB; }
[data-testid="stHeader"] { background:rgba(244,247,251,.96); }
.block-container {max-width:1450px; padding:3.1rem 2.7rem 4rem;}
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #DCE6F2; }
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stFileUploader > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzoneInstructions"] {
  color:#17324D !important;
}
[data-testid="stSidebar"] .stRadio label {padding:.22rem 0;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
  border-radius:12px; margin:4px 0; padding:10px 12px !important;
  background:#F7FAFD; border:1px solid #E3EBF5; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
  background:#EEF4FF; border-color:#C9D8F1; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-selected="true"] {
  background:#EAF2FF; border-color:#2B63DA; box-shadow:inset 3px 0 0 #2B63DA; }
[data-testid="stSidebar"] .stFileUploader section {
  background:#FFFFFF; border:1px dashed #BACBE0; border-radius:14px; }
[data-testid="stSidebar"] .stFileUploader section button {
  background:#17324D !important; color:#FFFFFF !important; border:none; }
[data-testid="stSidebar"] .stFileUploader section button *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
  color:#FFFFFF !important; opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
  color:#33516F !important; opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background:#F8FBFF; border-radius:14px; }
h1,h2,h3,h4 {font-family:'Plus Jakarta Sans',sans-serif !important;
 letter-spacing:-.035em !important; color:#102A4C !important;}
h1 {font-size:2rem!important;font-weight:800!important;}
h2 {font-size:1.35rem!important;font-weight:800!important;}
h3 {font-size:1.05rem!important;}
p {line-height:1.65; color:#304963;}
/* Match chart and all widgets to the light palette, including when the user has
   previously selected a dark theme in the browser. Scoped selectors avoid making
   native alert text invisible on a dark alert surface. */
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul,
[data-testid="stMarkdownContainer"] > ol,
[data-testid="stCaptionContainer"] p {
  color:#233F5B !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {
  color:#1B3957 !important; font-weight:650 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background:#FFFFFF !important; color:#193651 !important;
  border:1px solid #C6D6E8 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stMultiSelect"] [data-baseweb="select"] * {
  color:#193651 !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {color:#173755 !important;}
/* Alerts must be readable even when the Streamlit theme selected in the browser
   differs from the theme set by our repo. */
[data-testid="stAlert"] {background:#EAF2FF !important;border:1px solid #C7DDF6 !important;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span,
[data-testid="stAlert"] svg {color:#173F6C !important;fill:currentColor;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color:#173F6C !important;}
.hero {border-radius:20px;padding:32px 36px 33px;margin-bottom:24px;
 background:linear-gradient(112deg,#0D2849 0%,#183E78 58%,#346BD6 100%);
 box-shadow:0 16px 35px rgba(13,43,95,.13);position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-115px;right:-90px;width:400px;height:400px;
 border:1px solid #ffffff3a;border-radius:50%;box-shadow:0 0 0 65px #ffffff09,0 0 0 135px #ffffff07;}
.hero .eyebrow{font-size:.74rem;font-weight:800;letter-spacing:.16em;color:#D7E6FF;}
.hero .heading{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;
 font-size:1.94rem;line-height:1.3;color:white;letter-spacing:-.035em;margin-top:9px;}
.hero .lead{color:#E4EEFF;line-height:1.7;font-size:.95rem;max-width:860px;margin:12px 0 0;}
.kicker{font-size:.73rem;font-weight:800;letter-spacing:.14em;color:#2B63DA;margin:18px 0 8px;}
.stat-card{min-height:132px;background:#FFF;border:1px solid #DDE7F2;border-radius:16px;
 padding:20px 20px 18px;box-shadow:0 8px 24px rgba(20,52,92,.05);margin-bottom:12px;}
.stat-card .label{font-size:.82rem;color:#4D6783;font-weight:750;}
.stat-card .num{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.95rem;font-weight:800;
 color:#13345B;letter-spacing:-.055em;margin-top:10px;}
.stat-card .hint{font-size:.78rem;color:#556D87;margin-top:8px;line-height:1.45;}
.info-panel{border:1px solid #DBE6F2;background:#FFF;border-radius:14px;
 padding:19px 21px;margin:12px 0 17px;line-height:1.6;}
.info-panel .panel-header{font-family:'Plus Jakarta Sans',sans-serif;font-size:1rem;
 font-weight:800;color:#173755;margin-bottom:8px;}
.info-panel .panel-body{font-size:.88rem;color:#37516C;}
.callout{border-left:4px solid #336CD9;background:#EAF2FF;border-radius:0 12px 12px 0;
 padding:14px 18px;font-size:.90rem;line-height:1.65;color:#1F426D;margin:13px 0 17px;}
.warning{border-left-color:#D88A2A;background:#FFF6E8;color:#6B471E;}
.section-label{font-size:1.16rem;font-weight:800;color:#16385B;margin:14px 0 9px;}
.divider-caption{font-size:.84rem;color:#516982;margin:0 0 14px;}
[data-testid="stMetric"] {background:#fff;padding:16px 18px;border-radius:14px;border:1px solid #DDE7F2;}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {font-weight:700 !important; color:#39536D !important;}
[data-testid="stMetric"] [data-testid="stMetricValue"] {font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:800 !important; color:#173755 !important;}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {font-weight:700 !important; color:#1E6F58 !important;}
[data-testid="stDataFrame"] {border:1px solid #DDE7F2;border-radius:13px;overflow:hidden; background:#FFFFFF;}
[data-testid="stExpander"] {border-radius:12px;border:1px solid #DDE7F2;background:#fff;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{
 border-radius:10px;min-height:42px;font-weight:750; background:#17324D; color:#FFFFFF; border:none;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover {
  background:#102A4C; color:#FFFFFF; }
[data-testid="stTabs"] [data-baseweb="tab"] {font-weight:700; color:#21415F !important;}
[data-testid="stTabs"] [aria-selected="true"] {color:#173755 !important;}
.footer-note{color:#506984;font-size:.82rem;line-height:1.6;margin-top:22px;padding-top:10px;border-top:1px solid #DDE7F2;}
@media(max-width:900px){.block-container{padding:2.8rem 1rem 2rem}.hero{padding:23px}.hero .heading{font-size:1.47rem}}
</style>
""",
    unsafe_allow_html=True,
)


def hero(kicker: str, heading: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<div class="heading">{heading}</div><div class="lead">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def stat(label: str, number: str, hint: str) -> None:
    st.markdown(
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="num">{number}</div><div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def panel(title: str, content: str) -> None:
    st.markdown(
        f'<div class="info-panel"><div class="panel-header">{title}</div>'
        f'<div class="panel-body">{content}</div></div>',
        unsafe_allow_html=True,
    )


def subheading(title: str, description: str = "") -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="divider-caption">{description}</div>', unsafe_allow_html=True)


def base_chart(height: int = 355, margin: dict | None = None) -> dict:
    return dict(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=margin or dict(l=12, r=18, t=20, b=35),
        template="plotly_white",
        font=dict(family="DM Sans, sans-serif", size=13, color="#284560"),
        hoverlabel=dict(bgcolor="#FFF", font=dict(color="#173755", size=12)),
    )


def show_chart(fig: go.Figure) -> None:
    # Disable Streamlit's graph theme: otherwise a previously selected dark theme
    # can turn axis labels white on our explicitly light chart background.
    fig.update_layout(template="plotly_white", font=dict(color="#284560"),
                      legend=dict(font=dict(color="#284560", size=12)))
    fig.update_xaxes(tickfont=dict(color="#334F6B", size=12),
                     title_font=dict(color="#284560"))
    fig.update_yaxes(tickfont=dict(color="#334F6B", size=12),
                     title_font=dict(color="#284560"))
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displayModeBar": False})


def donut_counts(pass_count: int, fail_count: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Pass", "Fail"], values=[pass_count, fail_count], hole=.73,
        marker=dict(colors=[BLUE, CORAL]), sort=False, textinfo="percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="%{label}: %{value:,} wafer (%{percent})<extra></extra>",
    ))
    fig.update_layout(**base_chart(315, dict(l=0, r=0, t=10, b=10)),
                      showlegend=True, legend=dict(orientation="h", x=.25, y=-.09))
    fig.add_annotation(x=.5, y=.52, text=f"<b>{pass_count + fail_count:,}</b>",
                       showarrow=False, font=dict(size=29, color=NAVY))
    fig.add_annotation(x=.5, y=.42, text="total wafer", showarrow=False,
                       font=dict(size=11, color=MUTED))
    return fig


def df_models() -> pd.DataFrame:
    return pd.DataFrame(
        MODEL_RESULTS,
        columns=["Model", "PR-AUC", "SD PR-AUC", "ROC-AUC", "G-mean", "Akurasi"],
    )


def df_operating() -> pd.DataFrame:
    return pd.DataFrame(
        OPERATING_POINTS,
        columns=["Strategi", "Threshold", "TP", "FN", "FP", "TN", "Sensitivitas", "Spesifisitas", "Presisi", "G-mean", "MCC", "Akurasi"],
    )


def bar_models(data: pd.DataFrame, metric: str) -> go.Figure:
    d = data.sort_values(metric)
    colors = [BLUE if n == "Extra Trees" else "#9DB5DC" for n in d["Model"]]
    fig = go.Figure(go.Bar(
        y=d["Model"], x=d[metric], orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in d[metric]], textposition="outside",
        hovertemplate="%{y}<br>" + metric + ": %{x:.3f}<extra></extra>",
    ))
    maximum = max(1e-5, float(d[metric].max()))
    fig.update_layout(**base_chart(max(285, len(d)*37+55), dict(l=6,r=50,t=18,b=26)),
                      xaxis=dict(range=[0, min(1.0, maximum*1.25)], showgrid=True, gridcolor="#E5EDF6"),
                      yaxis=dict(title="", automargin=True), showlegend=False)
    return fig


def confusion_chart(row: pd.Series) -> go.Figure:
    # Rows are actual classes: Pass, Fail. Columns are predictions: Pass, Fail.
    matrix = [[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]]
    colors = [[0, "#EAF2FD"], [.5, "#97B8EB"], [1, "#2E69D4"]]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Prediksi Pass", "Prediksi Fail"],
        y=["Aktual Pass", "Aktual Fail"], colorscale=colors,
        zmin=0, zmax=max(1, max(map(max, matrix))), showscale=False,
        text=[[f"TN<br><b>{row['TN']}</b>", f"FP<br><b>{row['FP']}</b>"],
              [f"FN<br><b>{row['FN']}</b>", f"TP<br><b>{row['TP']}</b>"]],
        texttemplate="%{text}", textfont=dict(size=18),
        hovertemplate="%{y} → %{x}<br>%{z} wafer<extra></extra>",
    ))
    fig.update_layout(**base_chart(320, dict(l=5,r=10,t=16,b=42)),
                      xaxis=dict(side="bottom", tickfont=dict(size=11)),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=11)))
    return fig


@st.cache_data(show_spinner=False)
def read_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(raw_bytes), low_memory=False)
    if frame.empty:
        raise ValueError("File CSV tidak memiliki baris data.")
    return frame


def sensor_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"Time", "Pass/Fail"}
    data = frame.drop(columns=[k for k in excluded if k in frame.columns]).copy()
    # Field sensor SECOM numerik. Kolom teks yang tidak numerik menjadi NaN.
    return data.apply(pd.to_numeric, errors="coerce")


def raw_class_counts(frame: pd.DataFrame) -> tuple[int, int] | None:
    if "Pass/Fail" not in frame.columns:
        return None
    vals = pd.to_numeric(frame["Pass/Fail"], errors="coerce")
    if not vals.dropna().isin([-1, 1]).all():
        return None
    if vals.isna().any():
        return None
    return int((vals == -1).sum()), int((vals == 1).sum())


def month_series(frame: pd.DataFrame) -> pd.DataFrame | None:
    counts = raw_class_counts(frame)
    if counts is None or "Time" not in frame.columns:
        return None
    dt = pd.to_datetime(frame["Time"], errors="coerce")
    if dt.notna().sum() < 2:
        return None
    t = pd.DataFrame({"Time":dt, "Fail":(pd.to_numeric(frame["Pass/Fail"]) == 1).astype(int)})
    t = t.dropna(subset=["Time"])
    summary = t.groupby(t["Time"].dt.to_period("M"))["Fail"].agg(["size", "mean"])
    summary = summary.reset_index().rename(columns={"size":"Jumlah wafer", "mean":"Fail rate"})
    summary["Bulan"] = summary["Time"].astype(str)
    return summary


with st.sidebar:
    st.markdown("""
    <div style="padding:17px 4px 8px;display:flex;align-items:center;gap:12px;">
      <div style="border-radius:12px;padding:10px 12px;background:#EAF1FF;color:#2C63D2;font-size:24px;">◈</div>
      <div><div style="font-weight:800;font-size:1.04rem;color:#153656;">SECOM Insight</div>
      <div style="font-size:.72rem;color:#8493A7;">Wafer Quality Analytics</div></div>
    </div><hr style="border:0;border-top:1px solid #E8EDF5;margin:14px 0 9px;">
    """, unsafe_allow_html=True)
    st.caption("Tampilan v1.2 · Kontras diperbaiki")
    selection = st.radio(
        "NAVIGASI ANALISIS",
        ["📊 Ringkasan", "🧪 Kualitas Data", "🏆 Perbandingan Model",
         "🎯 Evaluasi & Threshold", "📈 Drift & Biaya", "🔬 Insight Sensor"],
        label_visibility="visible", key="secom_navigation",
    )
    st.divider()
    st.caption("SUMBER DATA INTERAKTIF (OPSIONAL)")
    upload = st.file_uploader("Unggah dataset SECOM (.csv)", type=["csv"],
                              help="Bisa menggunakan data_.csv. File hanya dipakai untuk eksplorasi dan tidak mengubah metrik hasil notebook.")
    df_upload = None
    file_path = Path("data_.csv")
    if upload is not None:
        try:
            df_upload = read_uploaded_csv(upload.getvalue())
            st.success(f"CSV terbaca: {len(df_upload):,} baris")
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")
    elif file_path.is_file():
        try:
            df_upload = read_uploaded_csv(file_path.read_bytes())
            st.success(f"Membaca data_.csv: {len(df_upload):,} baris")
        except Exception as exc:
            st.error(f"Gagal membuka data_.csv: {exc}")
    else:
        st.info("Dashboard tetap dapat dibuka tanpa dataset. Grafik hasil notebook sudah tersedia.")
    st.divider()
    st.caption("Sumber: notebook SECOM, hasil eksperimen yang sudah dijalankan. **Unggahan CSV tidak melatih ulang model.**")


if selection == "📊 Ringkasan":
    hero("OVERVIEW / SEMICONDUCTOR ANALYTICS", "Klasifikasi Kegagalan Wafer", "Ringkasan interaktif eksperimen SECOM: mendeteksi wafer berisiko gagal meskipun kelas Fail hanya sekitar 6,6% dari seluruh pengamatan.")
    a,b,c,d = st.columns(4, gap="medium")
    with a: stat("Total wafer", "1.567", "1.463 Pass · 104 Fail")
    with b: stat("Fitur sensor", "590", "Sinyal proses produksi anonim")
    with c: stat("Proporsi Fail", "6,64%", "Data sangat tidak seimbang")
    with d: stat("Model terpilih", "Extra Trees", "PR-AUC CV 0,223 ± 0,083")

    left,right = st.columns([.95,1.4], gap="large")
    with left:
        subheading("Distribusi status wafer", "Jumlah data asli: 1.567 wafer")
        show_chart(donut_counts(1463, 104))
    with right:
        subheading("Kinerja pada test set", "Extra Trees · test stratified 25% · threshold hasil OOF training")
        p,q,r = st.columns(3)
        p.metric("PR-AUC", "0,240", "CI95% 0,134–0,394")
        q.metric("ROC-AUC", "0,798", "CI95% 0,717–0,875")
        r.metric("Recall Fail", "80,8%", "21/26 Fail tertangkap")
        st.markdown('<div class="callout"><b>Mengapa akurasi tidak diutamakan?</b> Model yang selalu memprediksi Pass sudah memperoleh akurasi 93,4%, namun gagal mendeteksi seluruh wafer Fail. Oleh sebab itu, penelitian memprioritaskan PR-AUC dan sensitivitas.</div>', unsafe_allow_html=True)
        panel("Cara membaca hasil", "PR-AUC menunjukkan kemampuan mengurutkan kasus Fail pada kelas yang jarang. ROC-AUC mengukur kemampuan memisahkan Pass dan Fail. Recall menunjukkan berapa banyak wafer Fail yang berhasil terdeteksi.")

    subheading("Alur penelitian", "Prinsip validasi: tahap yang mempelajari data dilakukan hanya pada data training")
    steps = [
        ("01", "Dataset & kualitas", "1.567 wafer, 590 sensor; periksa missing value dan fitur konstan."),
        ("02", "Split & preprocessing", "Train/test 75:25 stratified; cleaning dan imputasi dalam fold."),
        ("03", "Seleksi model", "Screening 19 algoritma; pemilihan berdasarkan PR-AUC 5×2 CV."),
        ("04", "Threshold & evaluasi", "Threshold dari out-of-fold training; evaluasi sekali pada test set."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (idx, name, desc) in zip(cols, steps):
        with col:
            panel(f"<span style='color:#2B63DA'>{idx}</span> · {name}", desc)
    st.markdown('<div class="footer-note">Catatan: seluruh metrik pada halaman ini merupakan hasil eksperimen notebook, bukan hasil perhitungan ulang di dashboard.</div>', unsafe_allow_html=True)

elif selection == "🧪 Kualitas Data":
    hero("DATA PROFILING / SECOM", "Kualitas Data Sensor", "Pahami pola data hilang, fitur konstan, dan komposisi label sebelum melakukan pemodelan.")
    if df_upload is None:
        st.info("Saat ini menggunakan rekap kualitas data dari notebook. Unggah data_.csv di sidebar untuk memeriksa dataset secara langsung.")
        records = 1567; numeric_features = 590; missing = 4.5375; constant=116; missing_heavy=28
        class_counts = (1463,104)
        quality_df = None
    else:
        try:
            X = sensor_numeric(df_upload)
            records = len(df_upload); numeric_features = len(X.columns)
            missing = float(X.isna().to_numpy().mean()*100) if len(X.columns) else 0
            constant = int((X.nunique(dropna=True) <= 1).sum())
            missing_heavy = int((X.isna().mean() > .5).sum())
            class_counts = raw_class_counts(df_upload)
            quality_df = pd.DataFrame({"Sensor":X.columns.astype(str), "Persentase kosong":(X.isna().mean()*100).values,
                                      "Nilai unik":X.nunique(dropna=True).values})
        except Exception as exc:
            st.error(f"Gagal menghitung statistik data: {exc}")
            st.stop()
    a,b,c,d = st.columns(4, gap="medium")
    with a: stat("Jumlah wafer", f"{records:,}".replace(",","."), "Baris dataset")
    with b: stat("Sensor", str(numeric_features), "Kolom pengukuran")
    with c: stat("Missing cell", f"{missing:.2f}%".replace(".",","), "Proporsi sel yang kosong")
    with d: stat("Sensor konstan / kosong", str(constant), f"{missing_heavy} sensor missing > 50%")

    left, right = st.columns([1.05,1], gap="large")
    with left:
        subheading("Komposisi kelas", "Label -1 = Pass, 1 = Fail")
        if class_counts is None:
            st.warning("CSV tidak memiliki label Pass/Fail valid (-1 dan 1), sehingga distribusi kelas tidak dapat dihitung.")
        else:
            show_chart(donut_counts(*class_counts))
    with right:
        subheading("Pemeriksaan data", "Ringkasan kualitas penting dari notebook")
        if df_upload is None:
            st.dataframe(pd.DataFrame([
                ("Sensor yang punya missing value", 538),
                ("Sensor missing > 50%", 28),
                ("Sensor konstan", 116),
                ("Sensor dengan ≤ 2 nilai unik", 122),
                ("Wafer dengan minimal 1 nilai kosong", 1567),
                ("Baris duplikat", 0),
            ], columns=["Indikator", "Jumlah"]), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame([
                ("Sensor yang punya missing value", int((quality_df["Persentase kosong"]>0).sum())),
                ("Sensor missing > 50%", missing_heavy),
                ("Sensor konstan / kosong", constant),
                ("Wafer dengan minimal 1 nilai kosong", int(sensor_numeric(df_upload).isna().any(axis=1).sum())),
                ("Baris duplikat", int(df_upload.duplicated().sum())),
            ], columns=["Indikator", "Jumlah"]), use_container_width=True, hide_index=True)
        panel("Implikasi preprocessing", "Sensor dengan banyak missing value atau tanpa variasi dapat mengganggu model. Notebook membuang fitur yang terlalu kosong / konstan serta mengimputasi data menggunakan median, semuanya hanya berdasarkan training fold.")

    if df_upload is not None:
        st.divider()
        subheading("Eksplorasi interaktif dataset", "Grafik berikut dihitung dari CSV yang diunggah, bukan nilai tetap notebook")
        tab1, tab2, tab3 = st.tabs(["Missing value", "Sensor individual", "Pratinjau data"])
        with tab1:
            top_count = st.slider("Jumlah sensor dengan missing terbanyak", 5, min(40,max(5,numeric_features)),
                                  min(20,max(5,numeric_features)), key="missing_n") if numeric_features>=5 else 0
            data = quality_df.sort_values("Persentase kosong", ascending=False).head(top_count)
            if len(data):
                fig=go.Figure(go.Bar(x=data["Sensor"], y=data["Persentase kosong"],
                                     marker_color=BLUE, hovertemplate="%{x}: %{y:.1f}% kosong<extra></extra>"))
                fig.update_layout(**base_chart(360), yaxis_title="Missing value (%)",xaxis_title="Sensor")
                show_chart(fig)
            else:
                st.info("Tidak ada sensor untuk ditampilkan.")
        with tab2:
            numeric = sensor_numeric(df_upload)
            possible=[str(c) for c in numeric.columns if numeric[c].notna().sum() > 0]
            if not possible:
                st.warning("Tidak ditemukan sensor numerik yang memiliki nilai valid.")
            else:
                chosen = st.selectbox("Pilih sensor yang ingin diamati", possible)
                fig=go.Figure()
                counts = raw_class_counts(df_upload)
                vals=numeric[chosen]
                if counts is not None:
                    label=pd.to_numeric(df_upload["Pass/Fail"], errors="coerce")
                    for clas,name,color in [(-1,"Pass",BLUE),(1,"Fail",CORAL)]:
                        fig.add_histogram(x=vals[label==clas].dropna(), name=name,
                                          marker_color=color, opacity=.65, nbinsx=45)
                    fig.update_layout(barmode="overlay")
                else:
                    fig.add_histogram(x=vals.dropna(), name="Sensor",marker_color=BLUE,nbinsx=45)
                fig.update_layout(**base_chart(360), xaxis_title=chosen,yaxis_title="Jumlah wafer")
                show_chart(fig)
                st.caption("Perbandingan distribusi bersifat eksploratif, bukan bukti sebab akibat.")
        with tab3:
            cols=st.slider("Kolom yang ditampilkan", 5, min(len(df_upload.columns), 35),
                           min(12,len(df_upload.columns))) if len(df_upload.columns)>=5 else len(df_upload.columns)
            st.dataframe(df_upload.iloc[:100, :cols], use_container_width=True, hide_index=True)
            st.download_button("Unduh ringkasan kualitas sensor (CSV)",
                               data=quality_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="secom_kualitas_sensor.csv", mime="text/csv")

elif selection == "🏆 Perbandingan Model":
    hero("MODEL BENCHMARK / 5 × 2 CV", "Perbandingan 19 Algoritma", "Hasil screening cross-validation dari notebook. Bandingkan PR-AUC, ROC-AUC, G-mean, dan akurasi tanpa menjalankan training ulang.")
    a,b,c = st.columns(3)
    with a: stat("Model terbaik", "Extra Trees", "PR-AUC tertinggi pada 5×2 CV")
    with b: stat("PR-AUC CV", "0,223", "Simpangan baku ±0,083")
    with c: stat("Jumlah algoritma", "19", "+ baseline dan varian lanjutan di notebook")
    data = df_models()
    l,r = st.columns([1,1],gap="medium")
    with l:
        metric = st.selectbox("Urutkan berdasarkan metrik", ["PR-AUC","ROC-AUC","G-mean","Akurasi"], index=0)
    with r:
        n = st.slider("Jumlah model ditampilkan",5,19,10)
    ordered = data.sort_values(metric, ascending=False).head(n)
    subheading("Peringkat algoritma", "Model terbaik menurut metrik yang dipilih akan berada di bagian atas tabel")
    show_chart(bar_models(ordered, metric))
    st.dataframe(
        data.sort_values(metric,ascending=False).reset_index(drop=True).style.format({k:"{:.3f}" for k in ["PR-AUC","SD PR-AUC","ROC-AUC","G-mean","Akurasi"]}),
        use_container_width=True, hide_index=True,
    )
    st.download_button("Unduh seluruh hasil perbandingan (CSV)",
                       data=data.to_csv(index=False).encode("utf-8-sig"),
                       file_name="secom_perbandingan_19_model.csv", mime="text/csv")
    st.markdown('<div class="callout"><b>Perhatikan:</b> G-mean Extra Trees = 0,000 pada screening karena threshold default 0,50 hampir selalu memprediksi Pass. Setelah threshold dioptimalkan memakai out-of-fold training, sensitivitas test meningkat menjadi 80,8%. Nilai akurasi yang tinggi tidak otomatis berarti pendeteksian Fail baik.</div>', unsafe_allow_html=True)
    with st.expander("Apa yang dilakukan setelah screening 19 algoritma?"):
        st.markdown("Notebook juga membandingkan resampling, seleksi fitur, tuning Optuna, soft voting, dan stacking. PR-AUC CV **Extra Trees tuned = 0,217**, **stacking = 0,213**, dan **soft voting = 0,208**; hasil ini belum melampaui Extra Trees dasar (0,223). Perbedaan dengan pesaing teratas juga tidak signifikan pada uji pasangan terkoreksi.")

elif selection == "🎯 Evaluasi & Threshold":
    hero("TEST SET / OPERATING POINTS", "Evaluasi Model & Threshold", "Bandingkan trade-off antara wafer Fail yang tertangkap dan alarm palsu. Semua titik operasi ditetapkan pada training set, lalu dievaluasi pada test set.")
    ops = df_operating()
    name=st.selectbox("Strategi penentuan threshold",ops["Strategi"].tolist(),index=0)
    row=ops.loc[ops["Strategi"]==name].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    with c1: stat("Threshold",f"{row['Threshold']:.3f}".replace(".",","),"Ambang deteksi Fail")
    with c2: stat("Sensitivitas",f"{row['Sensitivitas']:.1%}".replace(".",","),f"{int(row['TP'])} dari 26 Fail terdeteksi")
    with c3: stat("Spesifisitas",f"{row['Spesifisitas']:.1%}".replace(".",","),f"{int(row['TN'])} dari 366 Pass aman")
    with c4: stat("Presisi",f"{row['Presisi']:.1%}".replace(".",","),f"{int(row['TP'])} dari {int(row['TP']+row['FP'])} alarm benar")
    left,right=st.columns([1.0,1.10],gap="large")
    with left:
        subheading("Confusion matrix", "Komposisi prediksi pada 392 wafer test")
        show_chart(confusion_chart(row))
        st.caption("TP: Fail benar terdeteksi · FN: Fail lolos · FP: Pass diberi alarm · TN: Pass benar.")
    with right:
        subheading("Evaluasi trade-off", "Mengubah threshold menukar sensitivitas dengan alarm palsu")
        fig=go.Figure()
        for key,title,color in [("Sensitivitas","Recall Fail",CORAL),("Spesifisitas","Spesifisitas",BLUE),("Presisi","Presisi",CYAN)]:
            fig.add_bar(x=[title],y=[row[key]],name=title,marker_color=color,
                        text=[f"{row[key]:.1%}"],textposition="outside")
        fig.update_layout(**base_chart(325), yaxis=dict(range=[0,1.1],tickformat=".0%",gridcolor="#E9EEF6"),
                          showlegend=False)
        show_chart(fig)
        panel("Makna praktis", "Pada threshold utama 0,137: 21 dari 26 wafer Fail berhasil dikenali, tetapi 107 wafer Pass menerima alarm palsu. Model cocok untuk <b>prioritas inspeksi tambahan</b>, bukan keputusan scrap otomatis.")
    subheading("Tabel alternatif threshold", "Pilihan di bawah berasal dari notebook (bukan simulasi threshold arbitrer)")
    st.dataframe(ops.style.format({"Threshold":"{:.3f}","Sensitivitas":"{:.1%}","Spesifisitas":"{:.1%}","Presisi":"{:.1%}","G-mean":"{:.3f}","MCC":"{:.3f}","Akurasi":"{:.1%}"}),
                 use_container_width=True,hide_index=True)
    st.markdown('<div class="callout warning"><b>Catatan interpretasi:</b> PR-AUC test = 0,240 dan ROC-AUC test = 0,798 tetap sama untuk semua titik operasi karena keduanya dihitung dari seluruh skor, bukan dari threshold tertentu. Metrik sensitivitas, spesifisitas, dan confusion matrix berubah ketika threshold berubah.</div>', unsafe_allow_html=True)

elif selection == "📈 Drift & Biaya":
    hero("ROBUSTNESS / RISK ANALYTICS", "Drift Temporal & Risiko Keputusan", "Hasil split acak terlihat menjanjikan, tetapi validasi kronologis mengungkap penurunan performa saat kondisi manufaktur berubah.")
    left,right=st.columns([1.04,1],gap="large")
    with left:
        subheading("Fail rate dari waktu ke waktu", "Angka bulanan yang dilaporkan notebook untuk Juli–Oktober 2008")
        month=pd.DataFrame(MONTHLY_FAIL,columns=["Bulan","Fail rate"])
        fig=go.Figure(go.Bar(x=month["Bulan"],y=month["Fail rate"],
                             marker_color=[CORAL,"#F1A3A9","#99B6E7","#6197DC"],
                             text=[f"{v:.1%}" for v in month["Fail rate"]],textposition="outside"))
        fig.update_layout(**base_chart(340),yaxis=dict(title="Fail rate",tickformat=".0%",range=[0,.27],gridcolor="#E5EDF6"))
        show_chart(fig)
        st.caption("Uji χ² antar-bulan pada notebook: χ² = 44,11; p < 0,001. Perbedaan fail rate menunjukkan dinamika proses yang tidak stabil.")
    with right:
        subheading("Validasi acak vs kronologis", "Performa model pada dua skema pengujian")
        drift=pd.DataFrame({"Metrik":["PR-AUC","ROC-AUC","Sensitivitas","G-mean"],
                            "Split acak":[.240,.798,.808,.756],
                            "Split kronologis":[.074,.570,.083,.270]})
        fig=go.Figure()
        fig.add_bar(name="Split acak",x=drift["Metrik"],y=drift["Split acak"],marker_color=BLUE,
                    text=[f"{v:.3f}" for v in drift["Split acak"]],textposition="outside")
        fig.add_bar(name="Split kronologis",x=drift["Metrik"],y=drift["Split kronologis"],marker_color=CORAL,
                    text=[f"{v:.3f}" for v in drift["Split kronologis"]],textposition="outside")
        fig.update_layout(**base_chart(340),barmode="group",yaxis=dict(range=[0,.93],gridcolor="#E5EDF6"),
                          legend=dict(orientation="h",y=-.2))
        show_chart(fig)
        st.markdown('<div class="callout warning"><b>Temuan utama:</b> PR-AUC turun dari 0,240 menjadi 0,074 pada split kronologis. Angka split acak kemungkinan optimistis untuk wafer produksi masa depan. Evaluasi drift dan retraining berkala diperlukan sebelum penerapan.</div>', unsafe_allow_html=True)

    st.divider()
    subheading("Simulasi biaya dari confusion matrix", "Biaya relatif FN (Fail terlewat) terhadap FP (alarm palsu), khusus titik operasi notebook")
    a,b=st.columns([1.05,1],gap="large")
    with a:
        ratio=st.slider("Rasio biaya FN : FP",min_value=1,max_value=50,value=10,step=1)
        comparison=df_operating()[["Strategi","Threshold","FP","FN"]].copy()
        comparison["Biaya per wafer"]=(ratio*comparison["FN"]+comparison["FP"])/392
        comparison=pd.concat([comparison,pd.DataFrame([
            {"Strategi":"Tanpa model: selalu Pass","Threshold":np.nan,"FP":0,"FN":26,"Biaya per wafer":ratio*26/392},
            {"Strategi":"Tanpa model: periksa semua","Threshold":np.nan,"FP":366,"FN":0,"Biaya per wafer":366/392},
        ])],ignore_index=True)
        st.dataframe(comparison[["Strategi","Biaya per wafer"]].sort_values("Biaya per wafer").style.format({"Biaya per wafer":"{:.3f}"}),
                     use_container_width=True,hide_index=True)
    with b:
        subheading("Bandingkan kebijakan", "Semakin kecil biaya per wafer, semakin efisien pada rasio tersebut")
        best=comparison.sort_values("Biaya per wafer").head(7)
        fig=go.Figure(go.Bar(y=best["Strategi"][::-1],x=best["Biaya per wafer"][::-1],orientation="h",
                             marker_color=[BLUE if "Tanpa model" not in x else "#ADC0D5" for x in best["Strategi"][::-1]],
                             text=[f"{v:.3f}" for v in best["Biaya per wafer"][::-1]],textposition="outside"))
        fig.update_layout(**base_chart(340,dict(l=5,r=53,t=10,b=20)),xaxis=dict(showgrid=True,gridcolor="#E7EEF7"),yaxis=dict(automargin=True))
        show_chart(fig)
        st.caption("Rumus: (rasio biaya FN × jumlah FN + jumlah FP) / 392 wafer. Tidak memperhitungkan biaya operasional lain.")
    st.markdown('<div class="callout"><b>Kesimpulan keputusan:</b> pemakaian model bergantung pada biaya nyata dari kesalahan. Model harus dipakai sebagai alat bantu prioritas inspeksi, dengan kontrol kualitas dan peninjauan berkala.</div>', unsafe_allow_html=True)

elif selection == "🔬 Insight Sensor":
    hero("INTERPRETABILITY / SENSOR SIGNALS", "Insight & Sensor Penting", "Temukan fitur yang paling konsisten relevan, pahami batas interpretasi 590 sensor anonim, dan pelajari risiko kebocoran data.")
    a,b,c=st.columns(3)
    with a: stat("Sensor diuji", "474", "Uji Mann–Whitney pada training set")
    with b: stat("Lolos FDR", "13", "Koreksi multiple testing")
    with c: stat("AUC tunggal maks.", "0,683", "Sinyal per sensor masih lemah")
    left,right=st.columns([1.18,1],gap="large")
    with left:
        subheading("Sensor kandidat teratas", "AUC absolut arah terbaik (max(AUC, 1−AUC)) untuk uji satu sensor")
        data=pd.DataFrame(TOP_SENSORS,columns=["Sensor","AUC tunggal","q-FDR"])
        data["AUC pembeda"] = data["AUC tunggal"].apply(lambda v:max(v,1-v))
        data=data.sort_values("AUC pembeda")
        fig=go.Figure(go.Bar(y=data["Sensor"],x=data["AUC pembeda"],orientation="h",
                             marker_color=BLUE,text=[f"{v:.3f}" for v in data["AUC pembeda"]],textposition="outside"))
        fig.update_layout(**base_chart(400,dict(l=12,r=47,t=15,b=24)),xaxis=dict(range=[0,.80],gridcolor="#E7EDF5"),yaxis=dict(automargin=True))
        show_chart(fig)
        st.caption("Angka berasal dari uji univariat di notebook, bukan feature importance multivariat.")
    with right:
        subheading("Sensor konsisten lintas metode", "Irisan SHAP, permutation importance, dan analisis univariat")
        for i in range(0,len(IMPORTANT_SENSORS),2):
            s1,s2=IMPORTANT_SENSORS[i:i+2]
            col1,col2=st.columns(2)
            col1.markdown(f'<div class="info-panel" style="padding:12px 15px;margin:4px 0"><div class="panel-header" style="margin:0">{s1}</div></div>',unsafe_allow_html=True)
            col2.markdown(f'<div class="info-panel" style="padding:12px 15px;margin:4px 0"><div class="panel-header" style="margin:0">{s2}</div></div>',unsafe_allow_html=True)
        panel("Batas interpretasi", "Nama sensor dalam dataset SECOM anonim. Penting secara statistik tidak otomatis berarti sebab kegagalan. Diperlukan metadata sensor dan validasi bersama ahli proses.")
    st.divider()
    subheading("Jebakan umum: data leakage", "Perbedaan nilai yang tampak bagus dan evaluasi yang benar")
    leaks=pd.DataFrame({"Pendekatan":["Preprocessing bocor", "Pipeline benar"],
                        "PR-AUC":[.998,.240],"Akurasi":[.981,.714]})
    fig=go.Figure()
    for metric,col in [("PR-AUC",CORAL),("Akurasi",BLUE)]:
        fig.add_bar(name=metric,x=leaks["Pendekatan"],y=leaks[metric],marker_color=col,
                    text=[f"{v:.3f}" for v in leaks[metric]],textposition="outside")
    fig.update_layout(**base_chart(320),barmode="group",yaxis=dict(range=[0,1.12],gridcolor="#E7EDF5"),
                      legend=dict(orientation="h",y=-.20))
    show_chart(fig)
    st.markdown('<div class="callout warning"><b>Hindari SMOTE, scaling, dan seleksi fitur sebelum train/test split.</b> Notebook memperlihatkan PR-AUC 0,998 dari pipeline yang bocor — angka tersebut tidak valid untuk generalisasi. Nilai PR-AUC 0,240 adalah evaluasi pada test set yang benar.</div>',unsafe_allow_html=True)

st.markdown('<div class="footer-note">SECOM Insight · Dashboard pembacaan hasil penelitian. Hasil model, threshold, dan grafik benchmark berasal dari notebook SECOM yang diberikan; CSV opsional hanya mengubah eksplorasi data. Untuk eksperimen/model baru, jalankan pipeline validasi notebook secara terpisah.</div>',unsafe_allow_html=True)
