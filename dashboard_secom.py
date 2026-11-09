"""SECOM Insight — dashboard interaktif hasil notebook klasifikasi wafer.

Jalankan: streamlit run dashboard_secom.py
Sumber metrik penelitian: notebook SECOM yang disampaikan pengguna (Nov 2026).
Angka benchmark statis berasal dari notebook. Unggah CSV hanya untuk EDA interaktif;
metrik benchmark TIDAK dihitung ulang dari CSV yang diunggah.
"""
from __future__ import annotations

import json
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
# Hasil penelitian berasal dari output Colab yang disertakan sebagai tabel
# NOTEBOOK_RESULTS di bawah. Tidak ada perhitungan ulang hasil model.
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



# Semua tabel di bawah diekstrak dari output notebook Google Colab yang diberikan pengguna.
NOTEBOOK_RESULTS = json.loads('{"split":[{"Keterangan":"Train","n":1175,"Fail":78,"Pass":1097,"proporsi Fail":0.0664},{"Keterangan":"Test","n":392,"Fail":26,"Pass":366,"proporsi Fail":0.0663}],"quality":[{"Keterangan":"Sel kosong (% dari seluruh sel)","Nilai":4.5375},{"Keterangan":"Fitur dengan missing > 0","Nilai":538.0},{"Keterangan":"Fitur dengan missing > 50%","Nilai":28.0},{"Keterangan":"Fitur konstan (1 nilai unik)","Nilai":116.0},{"Keterangan":"Fitur dengan ≤ 2 nilai unik","Nilai":122.0},{"Keterangan":"Wafer dengan ≥ 1 nilai kosong","Nilai":1567.0},{"Keterangan":"Baris duplikat","Nilai":0.0}],"baseline":[{"Keterangan":"Dummy: acak stratified","pr_auc":"0.074 ± 0.019","roc_auc":"0.498 ± 0.044","g_mean":"0.134 ± 0.177","bal_acc":"0.498 ± 0.044","mcc":"-0.005 ± 0.095","recall":"0.052 ± 0.082","specificity":"0.944 ± 0.006","accuracy":"0.885 ± 0.011","waktu (s)":1.8},{"Keterangan":"Dummy: selalu Pass","pr_auc":"0.066 ± 0.002","roc_auc":"0.500 ± 0.000","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":1.9}],"screening":[{"Keterangan":"Extra Trees","pr_auc":"0.223 ± 0.083","roc_auc":"0.749 ± 0.031","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":8.0},{"Keterangan":"Easy Ensemble","pr_auc":"0.219 ± 0.079","roc_auc":"0.712 ± 0.053","g_mean":"0.632 ± 0.053","bal_acc":"0.635 ± 0.050","mcc":"0.140 ± 0.051","recall":"0.615 ± 0.106","specificity":"0.655 ± 0.028","accuracy":"0.652 ± 0.025","waktu (s)":19.6},{"Keterangan":"Balanced Random Forest","pr_auc":"0.217 ± 0.071","roc_auc":"0.735 ± 0.034","g_mean":"0.469 ± 0.067","bal_acc":"0.584 ± 0.038","mcc":"0.158 ± 0.083","recall":"0.243 ± 0.068","specificity":"0.924 ± 0.033","accuracy":"0.879 ± 0.031","waktu (s)":8.8},{"Keterangan":"Random Forest","pr_auc":"0.211 ± 0.066","roc_auc":"0.733 ± 0.048","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":18.1},{"Keterangan":"XGBoost","pr_auc":"0.194 ± 0.061","roc_auc":"0.702 ± 0.042","g_mean":"0.107 ± 0.164","bal_acc":"0.517 ± 0.030","mcc":"0.082 ± 0.152","recall":"0.038 ± 0.059","specificity":"0.995 ± 0.005","accuracy":"0.932 ± 0.007","waktu (s)":17.6},{"Keterangan":"Logistic Regression (L1)","pr_auc":"0.189 ± 0.063","roc_auc":"0.689 ± 0.060","g_mean":"0.611 ± 0.078","bal_acc":"0.636 ± 0.058","mcc":"0.162 ± 0.070","recall":"0.480 ± 0.115","specificity":"0.791 ± 0.024","accuracy":"0.771 ± 0.023","waktu (s)":2.2},{"Keterangan":"LightGBM","pr_auc":"0.186 ± 0.050","roc_auc":"0.703 ± 0.045","g_mean":"0.025 ± 0.075","bal_acc":"0.503 ± 0.009","mcc":"0.016 ± 0.048","recall":"0.006 ± 0.019","specificity":"1.000 ± 0.001","accuracy":"0.934 ± 0.002","waktu (s)":15.4},{"Keterangan":"MLP + SMOTE","pr_auc":"0.185 ± 0.072","roc_auc":"0.670 ± 0.063","g_mean":"0.399 ± 0.168","bal_acc":"0.571 ± 0.061","mcc":"0.145 ± 0.131","recall":"0.199 ± 0.120","specificity":"0.943 ± 0.023","accuracy":"0.893 ± 0.023","waktu (s)":4.3},{"Keterangan":"CatBoost","pr_auc":"0.180 ± 0.041","roc_auc":"0.690 ± 0.046","g_mean":"0.156 ± 0.161","bal_acc":"0.520 ± 0.027","mcc":"0.072 ± 0.099","recall":"0.051 ± 0.056","specificity":"0.990 ± 0.005","accuracy":"0.928 ± 0.005","waktu (s)":45.4},{"Keterangan":"LDA (shrinkage)","pr_auc":"0.179 ± 0.061","roc_auc":"0.679 ± 0.046","g_mean":"0.552 ± 0.092","bal_acc":"0.607 ± 0.056","mcc":"0.141 ± 0.076","recall":"0.372 ± 0.109","specificity":"0.841 ± 0.025","accuracy":"0.810 ± 0.025","waktu (s)":2.1},{"Keterangan":"Logistic Regression (L2)","pr_auc":"0.179 ± 0.060","roc_auc":"0.668 ± 0.050","g_mean":"0.539 ± 0.091","bal_acc":"0.599 ± 0.055","mcc":"0.132 ± 0.072","recall":"0.353 ± 0.109","specificity":"0.845 ± 0.019","accuracy":"0.812 ± 0.019","waktu (s)":1.9},{"Keterangan":"Linear SVM","pr_auc":"0.178 ± 0.059","roc_auc":"0.667 ± 0.050","g_mean":"0.600 ± 0.059","bal_acc":"0.616 ± 0.044","mcc":"0.129 ± 0.049","recall":"0.493 ± 0.088","specificity":"0.738 ± 0.019","accuracy":"0.722 ± 0.018","waktu (s)":2.0},{"Keterangan":"SVM RBF","pr_auc":"0.172 ± 0.042","roc_auc":"0.667 ± 0.053","g_mean":"0.172 ± 0.145","bal_acc":"0.522 ± 0.024","mcc":"0.100 ± 0.113","recall":"0.051 ± 0.047","specificity":"0.992 ± 0.004","accuracy":"0.930 ± 0.005","waktu (s)":2.6},{"Keterangan":"HistGradientBoosting","pr_auc":"0.171 ± 0.041","roc_auc":"0.659 ± 0.047","g_mean":"0.050 ± 0.100","bal_acc":"0.505 ± 0.013","mcc":"0.042 ± 0.100","recall":"0.013 ± 0.025","specificity":"0.998 ± 0.003","accuracy":"0.933 ± 0.004","waktu (s)":28.3},{"Keterangan":"RUSBoost","pr_auc":"0.118 ± 0.034","roc_auc":"0.635 ± 0.076","g_mean":"0.426 ± 0.185","bal_acc":"0.555 ± 0.052","mcc":"0.068 ± 0.070","recall":"0.272 ± 0.171","specificity":"0.838 ± 0.081","accuracy":"0.801 ± 0.066","waktu (s)":2.1},{"Keterangan":"KNN (k=15) + SMOTE","pr_auc":"0.104 ± 0.026","roc_auc":"0.621 ± 0.086","g_mean":"0.324 ± 0.023","bal_acc":"0.519 ± 0.026","mcc":"0.029 ± 0.041","recall":"0.923 ± 0.049","specificity":"0.114 ± 0.014","accuracy":"0.168 ± 0.014","waktu (s)":2.3},{"Keterangan":"Decision Tree","pr_auc":"0.102 ± 0.048","roc_auc":"0.570 ± 0.056","g_mean":"0.502 ± 0.103","bal_acc":"0.571 ± 0.055","mcc":"0.090 ± 0.073","recall":"0.332 ± 0.136","specificity":"0.809 ± 0.073","accuracy":"0.778 ± 0.064","waktu (s)":2.8},{"Keterangan":"AdaBoost (stump berbobot)","pr_auc":"0.098 ± 0.015","roc_auc":"0.632 ± 0.047","g_mean":"0.501 ± 0.202","bal_acc":"0.589 ± 0.074","mcc":"0.095 ± 0.084","recall":"0.398 ± 0.225","specificity":"0.779 ± 0.088","accuracy":"0.754 ± 0.069","waktu (s)":2.6},{"Keterangan":"Gaussian Naive Bayes","pr_auc":"0.095 ± 0.020","roc_auc":"0.567 ± 0.061","g_mean":"0.513 ± 0.056","bal_acc":"0.543 ± 0.049","mcc":"0.045 ± 0.050","recall":"0.661 ± 0.136","specificity":"0.424 ± 0.142","accuracy":"0.440 ± 0.126","waktu (s)":1.9},{"Keterangan":"Dummy: acak stratified","pr_auc":"0.074 ± 0.019","roc_auc":"0.498 ± 0.044","g_mean":"0.134 ± 0.177","bal_acc":"0.498 ± 0.044","mcc":"-0.005 ± 0.095","recall":"0.052 ± 0.082","specificity":"0.944 ± 0.006","accuracy":"0.885 ± 0.011","waktu (s)":1.8},{"Keterangan":"Dummy: selalu Pass","pr_auc":"0.066 ± 0.002","roc_auc":"0.500 ± 0.000","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":1.9}],"resampling":[{"Keterangan":"ADASYN","LogReg · accuracy":0.82,"LogReg · g_mean":0.543,"LogReg · pr_auc":0.166,"LogReg · recall":0.353,"LogReg · roc_auc":0.66,"LogReg · specificity":0.854,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.076,"ExtraTrees · pr_auc":0.199,"ExtraTrees · recall":0.019,"ExtraTrees · roc_auc":0.706,"ExtraTrees · specificity":1.0},{"Keterangan":"Borderline-SMOTE","LogReg · accuracy":0.848,"LogReg · g_mean":0.484,"LogReg · pr_auc":0.163,"LogReg · recall":0.274,"LogReg · roc_auc":0.644,"LogReg · specificity":0.888,"ExtraTrees · accuracy":0.933,"ExtraTrees · g_mean":0.051,"ExtraTrees · pr_auc":0.2,"ExtraTrees · recall":0.013,"ExtraTrees · roc_auc":0.705,"ExtraTrees · specificity":0.999},{"Keterangan":"Class weight","LogReg · accuracy":0.812,"LogReg · g_mean":0.539,"LogReg · pr_auc":0.179,"LogReg · recall":0.353,"LogReg · roc_auc":0.668,"LogReg · specificity":0.845,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.0,"ExtraTrees · pr_auc":0.223,"ExtraTrees · recall":0.0,"ExtraTrees · roc_auc":0.749,"ExtraTrees · specificity":1.0},{"Keterangan":"Random over-sampling","LogReg · accuracy":0.823,"LogReg · g_mean":0.509,"LogReg · pr_auc":0.175,"LogReg · recall":0.308,"LogReg · roc_auc":0.664,"LogReg · specificity":0.86,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.0,"ExtraTrees · pr_auc":0.222,"ExtraTrees · recall":0.0,"ExtraTrees · roc_auc":0.753,"ExtraTrees · specificity":1.0},{"Keterangan":"Random under-sampling","LogReg · accuracy":0.653,"LogReg · g_mean":0.628,"LogReg · pr_auc":0.159,"LogReg · recall":0.603,"LogReg · roc_auc":0.661,"LogReg · specificity":0.656,"ExtraTrees · accuracy":0.682,"ExtraTrees · g_mean":0.664,"ExtraTrees · pr_auc":0.188,"ExtraTrees · recall":0.648,"ExtraTrees · roc_auc":0.713,"ExtraTrees · specificity":0.684},{"Keterangan":"SMOTE","LogReg · accuracy":0.829,"LogReg · g_mean":0.531,"LogReg · pr_auc":0.175,"LogReg · recall":0.333,"LogReg · roc_auc":0.657,"LogReg · specificity":0.864,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.076,"ExtraTrees · pr_auc":0.206,"ExtraTrees · recall":0.019,"ExtraTrees · roc_auc":0.722,"ExtraTrees · specificity":0.999},{"Keterangan":"SMOTE-ENN","LogReg · accuracy":0.38,"LogReg · g_mean":0.532,"LogReg · pr_auc":0.163,"LogReg · recall":0.82,"LogReg · roc_auc":0.65,"LogReg · specificity":0.348,"ExtraTrees · accuracy":0.531,"ExtraTrees · g_mean":0.6,"ExtraTrees · pr_auc":0.183,"ExtraTrees · recall":0.71,"ExtraTrees · roc_auc":0.665,"ExtraTrees · specificity":0.518},{"Keterangan":"SMOTE-Tomek","LogReg · accuracy":0.829,"LogReg · g_mean":0.531,"LogReg · pr_auc":0.175,"LogReg · recall":0.333,"LogReg · roc_auc":0.657,"LogReg · specificity":0.864,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.076,"ExtraTrees · pr_auc":0.206,"ExtraTrees · recall":0.019,"ExtraTrees · roc_auc":0.722,"ExtraTrees · specificity":0.999},{"Keterangan":"Tanpa penanganan","LogReg · accuracy":0.928,"LogReg · g_mean":0.061,"LogReg · pr_auc":0.189,"LogReg · recall":0.019,"LogReg · roc_auc":0.686,"LogReg · specificity":0.992,"ExtraTrees · accuracy":0.934,"ExtraTrees · g_mean":0.0,"ExtraTrees · pr_auc":0.218,"ExtraTrees · recall":0.0,"ExtraTrees · roc_auc":0.739,"ExtraTrees · specificity":1.0}],"feature_selection":[{"Keterangan":"ANOVA F (k=100)","LogReg · pr_auc":0.179,"LogReg · pr_auc_sd":0.049,"LogReg · roc_auc":0.668,"LogReg · time_s":1.757,"ExtraTrees · pr_auc":0.211,"ExtraTrees · pr_auc_sd":0.082,"ExtraTrees · roc_auc":0.74,"ExtraTrees · time_s":6.286},{"Keterangan":"ANOVA F (k=20)","LogReg · pr_auc":0.155,"LogReg · pr_auc_sd":0.061,"LogReg · roc_auc":0.665,"LogReg · time_s":1.927,"ExtraTrees · pr_auc":0.193,"ExtraTrees · pr_auc_sd":0.097,"ExtraTrees · roc_auc":0.688,"ExtraTrees · time_s":5.437},{"Keterangan":"ANOVA F (k=50)","LogReg · pr_auc":0.157,"LogReg · pr_auc_sd":0.049,"LogReg · roc_auc":0.661,"LogReg · time_s":1.849,"ExtraTrees · pr_auc":0.205,"ExtraTrees · pr_auc_sd":0.088,"ExtraTrees · roc_auc":0.72,"ExtraTrees · time_s":5.941},{"Keterangan":"L1-logistic (embedded)","LogReg · pr_auc":0.187,"LogReg · pr_auc_sd":0.061,"LogReg · roc_auc":0.683,"LogReg · time_s":2.195,"ExtraTrees · pr_auc":0.222,"ExtraTrees · pr_auc_sd":0.095,"ExtraTrees · roc_auc":0.741,"ExtraTrees · time_s":6.715},{"Keterangan":"Mutual information (k=50)","LogReg · pr_auc":0.146,"LogReg · pr_auc_sd":0.047,"LogReg · roc_auc":0.644,"LogReg · time_s":9.32,"ExtraTrees · pr_auc":0.181,"ExtraTrees · pr_auc_sd":0.056,"ExtraTrees · roc_auc":0.718,"ExtraTrees · time_s":13.604},{"Keterangan":"PCA (95% varians)","LogReg · pr_auc":0.173,"LogReg · pr_auc_sd":0.049,"LogReg · roc_auc":0.667,"LogReg · time_s":2.092,"ExtraTrees · pr_auc":0.137,"ExtraTrees · pr_auc_sd":0.029,"ExtraTrees · roc_auc":0.657,"ExtraTrees · time_s":7.283},{"Keterangan":"RF importance (top 50)","LogReg · pr_auc":0.167,"LogReg · pr_auc_sd":0.059,"LogReg · roc_auc":0.664,"LogReg · time_s":13.015,"ExtraTrees · pr_auc":0.196,"ExtraTrees · pr_auc_sd":0.068,"ExtraTrees · roc_auc":0.717,"ExtraTrees · time_s":18.146},{"Keterangan":"Semua fitur (setelah cleaning)","LogReg · pr_auc":0.179,"LogReg · pr_auc_sd":0.06,"LogReg · roc_auc":0.668,"LogReg · time_s":1.8,"ExtraTrees · pr_auc":0.223,"ExtraTrees · pr_auc_sd":0.083,"ExtraTrees · roc_auc":0.749,"ExtraTrees · time_s":7.699}],"top_models":[{"Keterangan":"Extra Trees","pr_auc":"0.223 ± 0.083","roc_auc":"0.749 ± 0.031","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":8.0},{"Keterangan":"Easy Ensemble","pr_auc":"0.219 ± 0.079","roc_auc":"0.712 ± 0.053","g_mean":"0.632 ± 0.053","bal_acc":"0.635 ± 0.050","mcc":"0.140 ± 0.051","recall":"0.615 ± 0.106","specificity":"0.655 ± 0.028","accuracy":"0.652 ± 0.025","waktu (s)":19.6},{"Keterangan":"Balanced Random Forest","pr_auc":"0.217 ± 0.071","roc_auc":"0.735 ± 0.034","g_mean":"0.469 ± 0.067","bal_acc":"0.584 ± 0.038","mcc":"0.158 ± 0.083","recall":"0.243 ± 0.068","specificity":"0.924 ± 0.033","accuracy":"0.879 ± 0.031","waktu (s)":8.8},{"Keterangan":"Extra Trees (tuned)","pr_auc":"0.217 ± 0.076","roc_auc":"0.746 ± 0.043","g_mean":"0.000 ± 0.000","bal_acc":"0.499 ± 0.001","mcc":"-0.004 ± 0.008","recall":"0.000 ± 0.000","specificity":"0.999 ± 0.003","accuracy":"0.932 ± 0.003","waktu (s)":9.2},{"Keterangan":"Stacking (ET+LGBM+LR → LR)","pr_auc":"0.213 ± 0.067","roc_auc":"0.710 ± 0.047","g_mean":"0.629 ± 0.043","bal_acc":"0.633 ± 0.038","mcc":"0.141 ± 0.040","recall":"0.583 ± 0.084","specificity":"0.683 ± 0.037","accuracy":"0.677 ± 0.032","waktu (s)":48.0},{"Keterangan":"Random Forest","pr_auc":"0.211 ± 0.066","roc_auc":"0.733 ± 0.048","g_mean":"0.000 ± 0.000","bal_acc":"0.500 ± 0.000","mcc":"0.000 ± 0.000","recall":"0.000 ± 0.000","specificity":"1.000 ± 0.000","accuracy":"0.934 ± 0.002","waktu (s)":18.1},{"Keterangan":"Soft Voting (ET+LGBM+CatBoost+LR)","pr_auc":"0.208 ± 0.060","roc_auc":"0.712 ± 0.039","g_mean":"0.202 ± 0.217","bal_acc":"0.538 ± 0.054","mcc":"0.107 ± 0.149","recall":"0.090 ± 0.113","specificity":"0.987 ± 0.006","accuracy":"0.928 ± 0.006","waktu (s)":60.9},{"Keterangan":"LightGBM (tuned)","pr_auc":"0.207 ± 0.071","roc_auc":"0.702 ± 0.041","g_mean":"0.229 ± 0.190","bal_acc":"0.539 ± 0.038","mcc":"0.130 ± 0.133","recall":"0.090 ± 0.077","specificity":"0.988 ± 0.004","accuracy":"0.929 ± 0.005","waktu (s)":7.2},{"Keterangan":"XGBoost","pr_auc":"0.194 ± 0.061","roc_auc":"0.702 ± 0.042","g_mean":"0.107 ± 0.164","bal_acc":"0.517 ± 0.030","mcc":"0.082 ± 0.152","recall":"0.038 ± 0.059","specificity":"0.995 ± 0.005","accuracy":"0.932 ± 0.007","waktu (s)":17.6},{"Keterangan":"Logistic Regression (L1)","pr_auc":"0.189 ± 0.063","roc_auc":"0.689 ± 0.060","g_mean":"0.611 ± 0.078","bal_acc":"0.636 ± 0.058","mcc":"0.162 ± 0.070","recall":"0.480 ± 0.115","specificity":"0.791 ± 0.024","accuracy":"0.771 ± 0.023","waktu (s)":2.2}],"significance":[{"Keterangan":"Easy Ensemble","Δ PR-AUC":0.0033,"t terkoreksi":0.0709,"p":0.945},{"Keterangan":"Balanced Random Forest","Δ PR-AUC":0.0054,"t terkoreksi":0.3125,"p":0.7618},{"Keterangan":"Extra Trees (tuned)","Δ PR-AUC":0.0055,"t terkoreksi":0.3318,"p":0.7477},{"Keterangan":"Stacking (ET+LGBM+LR → LR)","Δ PR-AUC":0.0096,"t terkoreksi":0.3317,"p":0.7477},{"Keterangan":"Random Forest","Δ PR-AUC":0.0119,"t terkoreksi":0.3668,"p":0.7223},{"Keterangan":"Soft Voting (ET+LGBM+CatBoost+LR)","Δ PR-AUC":0.0144,"t terkoreksi":0.467,"p":0.6516},{"Keterangan":"LightGBM (tuned)","Δ PR-AUC":0.0155,"t terkoreksi":0.4342,"p":0.6744},{"Keterangan":"Logistic Regression (L2)","Δ PR-AUC":0.0437,"t terkoreksi":0.8224,"p":0.4321}],"threshold_oof_train":[{"Keterangan":"G-mean maks. (UTAMA)","threshold":0.137,"sensitivity":0.654,"specificity":0.727,"precision":0.146,"g_mean":0.69,"bal_acc":0.691,"mcc":0.208,"f2":0.385,"accuracy":0.723},{"Keterangan":"Balanced acc. / Youden maks.","threshold":0.137,"sensitivity":0.654,"specificity":0.727,"precision":0.146,"g_mean":0.69,"bal_acc":0.691,"mcc":0.208,"f2":0.385,"accuracy":0.723},{"Keterangan":"MCC maks.","threshold":0.193,"sensitivity":0.372,"specificity":0.923,"precision":0.257,"g_mean":0.586,"bal_acc":0.648,"mcc":0.249,"f2":0.341,"accuracy":0.887},{"Keterangan":"F2 maks.","threshold":0.15,"sensitivity":0.59,"specificity":0.787,"precision":0.164,"g_mean":0.681,"bal_acc":0.688,"mcc":0.22,"f2":0.389,"accuracy":0.774},{"Keterangan":"Sens. maks. dgn akurasi ≥ 90%","threshold":0.217,"sensitivity":0.244,"specificity":0.953,"precision":0.268,"g_mean":0.482,"bal_acc":0.598,"mcc":0.205,"f2":0.248,"accuracy":0.906},{"Keterangan":"Default","threshold":0.5,"sensitivity":0.0,"specificity":1.0,"precision":0.0,"g_mean":0.0,"bal_acc":0.5,"mcc":0.0,"f2":0.0,"accuracy":0.934}],"test_ci":[{"Keterangan":"pr_auc","estimasi":0.24,"CI95_bawah":0.134,"CI95_atas":0.394,"baseline \\"selalu Pass\\"":0.066},{"Keterangan":"roc_auc","estimasi":0.798,"CI95_bawah":0.717,"CI95_atas":0.875,"baseline \\"selalu Pass\\"":0.5},{"Keterangan":"sensitivity","estimasi":0.808,"CI95_bawah":0.654,"CI95_atas":0.955,"baseline \\"selalu Pass\\"":0.0},{"Keterangan":"specificity","estimasi":0.708,"CI95_bawah":0.663,"CI95_atas":0.753,"baseline \\"selalu Pass\\"":1.0},{"Keterangan":"precision","estimasi":0.164,"CI95_bawah":0.102,"CI95_atas":0.232,"baseline \\"selalu Pass\\"":0.0},{"Keterangan":"g_mean","estimasi":0.756,"CI95_bawah":0.678,"CI95_atas":0.824,"baseline \\"selalu Pass\\"":0.0},{"Keterangan":"bal_acc","estimasi":0.758,"CI95_bawah":0.678,"CI95_atas":0.832,"baseline \\"selalu Pass\\"":0.5},{"Keterangan":"mcc","estimasi":0.273,"CI95_bawah":0.179,"CI95_atas":0.362,"baseline \\"selalu Pass\\"":0.0},{"Keterangan":"f2","estimasi":0.453,"CI95_bawah":0.327,"CI95_atas":0.562,"baseline \\"selalu Pass\\"":0.0},{"Keterangan":"accuracy","estimasi":0.714,"CI95_bawah":0.671,"CI95_atas":0.758,"baseline \\"selalu Pass\\"":0.934}],"threshold_test":[{"Keterangan":"G-mean maks. (UTAMA)","threshold":0.137,"TP":21.0,"FN":5.0,"FP":107.0,"TN":259.0,"sensitivity":0.808,"specificity":0.708,"precision":0.164,"NPV":0.981,"g_mean":0.756,"bal_acc":0.758,"mcc":0.273,"f2":0.453,"accuracy":0.714},{"Keterangan":"Balanced acc. / Youden maks.","threshold":0.137,"TP":21.0,"FN":5.0,"FP":107.0,"TN":259.0,"sensitivity":0.808,"specificity":0.708,"precision":0.164,"NPV":0.981,"g_mean":0.756,"bal_acc":0.758,"mcc":0.273,"f2":0.453,"accuracy":0.714},{"Keterangan":"MCC maks.","threshold":0.193,"TP":6.0,"FN":20.0,"FP":31.0,"TN":335.0,"sensitivity":0.231,"specificity":0.915,"precision":0.162,"NPV":0.944,"g_mean":0.46,"bal_acc":0.573,"mcc":0.124,"f2":0.213,"accuracy":0.87},{"Keterangan":"F2 maks.","threshold":0.15,"TP":19.0,"FN":7.0,"FP":76.0,"TN":290.0,"sensitivity":0.731,"specificity":0.792,"precision":0.2,"NPV":0.976,"g_mean":0.761,"bal_acc":0.762,"mcc":0.304,"f2":0.477,"accuracy":0.788},{"Keterangan":"Sens. maks. dgn akurasi ≥ 90%","threshold":0.217,"TP":5.0,"FN":21.0,"FP":17.0,"TN":349.0,"sensitivity":0.192,"specificity":0.954,"precision":0.227,"NPV":0.943,"g_mean":0.428,"bal_acc":0.573,"mcc":0.158,"f2":0.198,"accuracy":0.903},{"Keterangan":"Default","threshold":0.5,"TP":0.0,"FN":26.0,"FP":0.0,"TN":366.0,"sensitivity":0.0,"specificity":1.0,"precision":0.0,"NPV":0.934,"g_mean":0.0,"bal_acc":0.5,"mcc":0.0,"f2":0.0,"accuracy":0.934}],"cost":[{"Keterangan":"5 : 1","threshold":0.193,"biaya/wafer model (test)":0.334,"biaya/wafer \\"selalu Pass\\"":0.332,"biaya/wafer \\"periksa semua\\"":0.934,"sensitivitas":0.231,"spesifisitas":0.915,"penghematan vs strategi terbaik tanpa model":-0.008},{"Keterangan":"10 : 1","threshold":0.15,"biaya/wafer model (test)":0.372,"biaya/wafer \\"selalu Pass\\"":0.663,"biaya/wafer \\"periksa semua\\"":0.934,"sensitivitas":0.731,"spesifisitas":0.792,"penghematan vs strategi terbaik tanpa model":0.438},{"Keterangan":"20 : 1","threshold":0.105,"biaya/wafer model (test)":0.617,"biaya/wafer \\"selalu Pass\\"":1.327,"biaya/wafer \\"periksa semua\\"":0.934,"sensitivitas":0.885,"spesifisitas":0.503,"penghematan vs strategi terbaik tanpa model":0.339},{"Keterangan":"50 : 1","threshold":0.079,"biaya/wafer model (test)":0.844,"biaya/wafer \\"selalu Pass\\"":3.316,"biaya/wafer \\"periksa semua\\"":0.934,"sensitivitas":0.962,"spesifisitas":0.232,"penghematan vs strategi terbaik tanpa model":0.096}],"chronological":[{"Keterangan":"pr_auc","Split acak stratified":0.24,"Split kronologis":0.074},{"Keterangan":"roc_auc","Split acak stratified":0.798,"Split kronologis":0.57},{"Keterangan":"sensitivity","Split acak stratified":0.808,"Split kronologis":0.083},{"Keterangan":"specificity","Split acak stratified":0.708,"Split kronologis":0.872},{"Keterangan":"g_mean","Split acak stratified":0.756,"Split kronologis":0.27},{"Keterangan":"bal_acc","Split acak stratified":0.758,"Split kronologis":0.478},{"Keterangan":"mcc","Split acak stratified":0.273,"Split kronologis":-0.032},{"Keterangan":"accuracy","Split acak stratified":0.714,"Split kronologis":0.824},{"Keterangan":"prevalensi Fail test","Split acak stratified":0.066,"Split kronologis":0.061}],"shap":[{"Keterangan":"f59","mean|SHAP|":0.2501,"perm_ΔPR-AUC":0.0566,"perm_sd":0.0127,"rank_SHAP":1,"rank_perm":1,"rank_MWU":2.0},{"Keterangan":"f33","mean|SHAP|":0.2477,"perm_ΔPR-AUC":0.0248,"perm_sd":0.0191,"rank_SHAP":2,"rank_perm":2,"rank_MWU":11.0},{"Keterangan":"f103","mean|SHAP|":0.1944,"perm_ΔPR-AUC":0.0156,"perm_sd":0.0078,"rank_SHAP":3,"rank_perm":4,"rank_MWU":1.0},{"Keterangan":"f205","mean|SHAP|":0.1703,"perm_ΔPR-AUC":0.0219,"perm_sd":0.0154,"rank_SHAP":4,"rank_perm":3,"rank_MWU":7.0},{"Keterangan":"f21","mean|SHAP|":0.1703,"perm_ΔPR-AUC":0.0081,"perm_sd":0.0102,"rank_SHAP":5,"rank_perm":11,"rank_MWU":10.0},{"Keterangan":"f130","mean|SHAP|":0.1463,"perm_ΔPR-AUC":0.01,"perm_sd":0.0132,"rank_SHAP":6,"rank_perm":7,"rank_MWU":73.0},{"Keterangan":"f213","mean|SHAP|":0.1422,"perm_ΔPR-AUC":0.0062,"perm_sd":0.015,"rank_SHAP":7,"rank_perm":15,"rank_MWU":37.0},{"Keterangan":"f183","mean|SHAP|":0.1291,"perm_ΔPR-AUC":-0.0034,"perm_sd":0.0124,"rank_SHAP":8,"rank_perm":262,"rank_MWU":30.0},{"Keterangan":"f175","mean|SHAP|":0.1267,"perm_ΔPR-AUC":-0.0157,"perm_sd":0.0105,"rank_SHAP":9,"rank_perm":269,"rank_MWU":31.0},{"Keterangan":"f81","mean|SHAP|":0.12,"perm_ΔPR-AUC":-0.0083,"perm_sd":0.0061,"rank_SHAP":10,"rank_perm":266,"rank_MWU":61.0},{"Keterangan":"f247","mean|SHAP|":0.1171,"perm_ΔPR-AUC":0.0082,"perm_sd":0.0116,"rank_SHAP":11,"rank_perm":10,"rank_MWU":3.0},{"Keterangan":"f488","mean|SHAP|":0.1076,"perm_ΔPR-AUC":0.001,"perm_sd":0.008,"rank_SHAP":12,"rank_perm":91,"rank_MWU":114.0},{"Keterangan":"f31","mean|SHAP|":0.1004,"perm_ΔPR-AUC":0.0009,"perm_sd":0.0079,"rank_SHAP":13,"rank_perm":98,"rank_MWU":297.0},{"Keterangan":"f86","mean|SHAP|":0.1002,"perm_ΔPR-AUC":-0.0089,"perm_sd":0.0053,"rank_SHAP":14,"rank_perm":267,"rank_MWU":115.0},{"Keterangan":"f587","mean|SHAP|":0.0938,"perm_ΔPR-AUC":0.0131,"perm_sd":0.0074,"rank_SHAP":15,"rank_perm":5,"rank_MWU":62.0}],"anomaly":[{"Keterangan":"Hotelling T² (PCA)","ROC-AUC":0.573,"PR-AUC":0.091,"sensitivitas @batas":0.038,"false alarm @batas":0.066,"G-mean @batas":0.19},{"Keterangan":"SPE / Q (PCA)","ROC-AUC":0.604,"PR-AUC":0.103,"sensitivitas @batas":0.115,"false alarm @batas":0.085,"G-mean @batas":0.325},{"Keterangan":"T² ATAU SPE","ROC-AUC":0.613,"PR-AUC":0.1,"sensitivitas @batas":0.115,"false alarm @batas":0.107,"G-mean @batas":0.321},{"Keterangan":"Isolation Forest","ROC-AUC":0.612,"PR-AUC":0.142,"sensitivitas @batas":0.077,"false alarm @batas":0.003,"G-mean @batas":0.277},{"Keterangan":"LOF (novelty)","ROC-AUC":0.605,"PR-AUC":0.107,"sensitivitas @batas":0.0,"false alarm @batas":0.019,"G-mean @batas":0.0},{"Keterangan":"Supervised: Extra Trees","ROC-AUC":0.798,"PR-AUC":0.24,"sensitivitas @batas":0.808,"false alarm @batas":0.292,"G-mean @batas":0.756}],"leakage":[{"Keterangan":"pr_auc","❌ Bocor (SMOTE + seleksi sebelum split)":0.998,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.24},{"Keterangan":"roc_auc","❌ Bocor (SMOTE + seleksi sebelum split)":0.998,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.798},{"Keterangan":"sensitivity","❌ Bocor (SMOTE + seleksi sebelum split)":0.986,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.808},{"Keterangan":"specificity","❌ Bocor (SMOTE + seleksi sebelum split)":0.975,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.708},{"Keterangan":"g_mean","❌ Bocor (SMOTE + seleksi sebelum split)":0.981,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.756},{"Keterangan":"bal_acc","❌ Bocor (SMOTE + seleksi sebelum split)":0.981,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.758},{"Keterangan":"mcc","❌ Bocor (SMOTE + seleksi sebelum split)":0.962,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.273},{"Keterangan":"accuracy","❌ Bocor (SMOTE + seleksi sebelum split)":0.981,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.714},{"Keterangan":"prevalensi Fail di test","❌ Bocor (SMOTE + seleksi sebelum split)":0.5,"✅ Benar (pipeline, G-mean maks. (UTAMA))":0.066}],"summary":[{"Keterangan":"Model final","Nilai":"Extra Trees"},{"Keterangan":"PR-AUC CV (5×2)","Nilai":"0.223 ± 0.083"},{"Keterangan":"PR-AUC test [CI95]","Nilai":"0.240 [0.134, 0.394]"},{"Keterangan":"ROC-AUC test [CI95]","Nilai":"0.798 [0.717, 0.875]"},{"Keterangan":"Sensitivitas test [CI95]","Nilai":"0.808 [0.654, 0.955]"},{"Keterangan":"Spesifisitas test [CI95]","Nilai":"0.708 [0.663, 0.753]"},{"Keterangan":"G-mean test [CI95]","Nilai":"0.756 [0.678, 0.824]"},{"Keterangan":"MCC test [CI95]","Nilai":"0.273 [0.179, 0.362]"},{"Keterangan":"Akurasi test (NIR)","Nilai":"0.714 (0.934)"}],"sensor_univariate":[{"Keterangan":"f103","p_MWU":0.0,"AUC_tunggal":0.6779,"|AUC−0.5|":0.1779,"q_FDR":0.0001},{"Keterangan":"f59","p_MWU":0.0,"AUC_tunggal":0.6694,"|AUC−0.5|":0.1694,"q_FDR":0.0001},{"Keterangan":"f247","p_MWU":0.0,"AUC_tunggal":0.6699,"|AUC−0.5|":0.1699,"q_FDR":0.005},{"Keterangan":"f510","p_MWU":0.0,"AUC_tunggal":0.6375,"|AUC−0.5|":0.1375,"q_FDR":0.005},{"Keterangan":"f519","p_MWU":0.0001,"AUC_tunggal":0.6687,"|AUC−0.5|":0.1687,"q_FDR":0.005},{"Keterangan":"f477","p_MWU":0.0001,"AUC_tunggal":0.6354,"|AUC−0.5|":0.1354,"q_FDR":0.005},{"Keterangan":"f205","p_MWU":0.0001,"AUC_tunggal":0.6332,"|AUC−0.5|":0.1332,"q_FDR":0.0056},{"Keterangan":"f341","p_MWU":0.0003,"AUC_tunggal":0.6236,"|AUC−0.5|":0.1236,"q_FDR":0.0155},{"Keterangan":"f95","p_MWU":0.0005,"AUC_tunggal":0.6112,"|AUC−0.5|":0.1112,"q_FDR":0.0233},{"Keterangan":"f21","p_MWU":0.0005,"AUC_tunggal":0.6179,"|AUC−0.5|":0.1179,"q_FDR":0.0233},{"Keterangan":"f33","p_MWU":0.0006,"AUC_tunggal":0.6167,"|AUC−0.5|":0.1167,"q_FDR":0.0243},{"Keterangan":"f385","p_MWU":0.0008,"AUC_tunggal":0.6395,"|AUC−0.5|":0.1395,"q_FDR":0.033}]}')



def nb_table(key: str) -> pd.DataFrame:
    return pd.DataFrame(NOTEBOOK_RESULTS[key].copy())


def display_nb_table(key: str, *, title: str = '', help_text: str = '', height: int = 430) -> None:
    if title: subheading(title, help_text)
    st.dataframe(nb_table(key), use_container_width=True, hide_index=True, height=height)
    with st.expander('Sumber angka & unduh tabel ini', expanded=False):
        st.caption('Nilai tabel disalin dari OUTPUT notebook Google Colab (bukan dihitung ulang saat dashboard dibuka).')
        st.download_button('Unduh tabel sesuai Colab (.csv)',
                           data=nb_table(key).to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'secom_colab_{key}.csv', mime='text/csv', key=f'csv_{key}')


def cv_mean(text):
    try:
        return float(str(text).split('±')[0].strip())
    except (TypeError,ValueError):
        return float('nan')


def df_models() -> pd.DataFrame:
    df=nb_table('screening')
    df=df[~df['Keterangan'].astype(str).str.startswith('Dummy:')].copy()
    df=df.rename(columns={'Keterangan':'Model'})
    # The 19 reported classifier scores remain the same as in Google Colab.
    return df


def df_operating() -> pd.DataFrame:
    return nb_table('threshold_test').rename(columns={
        'Keterangan':'Strategi','threshold':'Threshold',
        'sensitivity':'Sensitivitas','specificity':'Spesifisitas',
        'precision':'Presisi','g_mean':'G-mean','bal_acc':'Balanced accuracy',
        'mcc':'MCC','f2':'F2','accuracy':'Akurasi',
    })


def nb_metric(key: str) -> dict:
    ci=nb_table('test_ci')
    return ci.set_index('Keterangan').loc[key].to_dict()


def plot_horizontal(df:pd.DataFrame,label_col:str,value_col:str,selected:str|None=None,height:int|None=None) -> None:
    d=df.sort_values(value_col)
    colors=[BLUE if label==selected else '#86AADF' for label in d[label_col]]
    fig=go.Figure(go.Bar(y=d[label_col],x=d[value_col],orientation='h',
                         marker_color=colors,text=[f'{float(x):.3f}' for x in d[value_col]],
                         textposition='outside'))
    mx=max(0.001,float(d[value_col].max()))
    fig.update_layout(**base_chart(height or max(300,len(d)*37+65),dict(l=10,r=55,t=15,b=32)),
                      xaxis=dict(range=[0,min(1.05,mx*1.30)],gridcolor='#E5EDF6'),
                      yaxis=dict(automargin=True),showlegend=False)
    show_chart(fig)


NAV=[
    '📊 Ringkasan',
    '🧪 Kualitas & Preprocessing',
    '🏆 Perbandingan Model',
    '⚙️ Resampling & Tuning',
    '🎯 Evaluasi & Threshold',
    '📈 Drift & Biaya',
    '🔬 Sensor & Anomali',
    '📚 Kesimpulan & Sumber',
]

with st.sidebar:
    st.markdown('''
    <div style="padding:17px 4px 8px;display:flex;align-items:center;gap:12px;">
      <div style="border-radius:12px;padding:10px 12px;background:#EAF1FF;color:#2C63D2;font-size:24px;">◈</div>
      <div><div style="font-weight:800;font-size:1.04rem;color:#153656;">SECOM Insight</div>
      <div style="font-size:.72rem;color:#506984;">Wafer Quality Analytics</div></div>
    </div><hr style="border:0;border-top:1px solid #E8EDF5;margin:14px 0 9px;">
    ''',unsafe_allow_html=True)
    st.caption('VERSI AKADEMIK · OUTPUT GOOGLE COLAB')
    selection=st.radio('NAVIGASI ANALISIS',NAV,key='secom_navigation')
    st.divider()
    st.caption('DATA SECOM (OPSIONAL)')
    upload=st.file_uploader('Unggah data_.csv untuk eksplorasi',type=['csv'],
                            help='Hanya mengubah EDA dataset yang diunggah; seluruh hasil eksperimen tetap sesuai output notebook.')
    df_upload=None
    local_csv=Path('data_.csv')
    if upload is not None:
        try:
            df_upload=read_uploaded_csv(upload.getvalue())
            st.success(f'CSV terbaca: {len(df_upload):,} baris')
        except Exception as exc:st.error(f'Gagal membaca CSV: {exc}')
    elif local_csv.is_file():
        try:
            df_upload=read_uploaded_csv(local_csv.read_bytes())
            st.success(f'Data lokal terbaca: {len(df_upload):,} baris')
        except Exception as exc:st.error(f'Gagal membaca data_.csv: {exc}')
    else:
        st.info('Tidak perlu upload CSV untuk melihat hasil Google Colab.')
    st.divider()
    st.caption('Versi 2.0 · Sinkron dengan output Google Colab')
    st.caption('Angka training, CV, OOF, dan test tidak dicampur. Hasil model merupakan hasil notebook tersimpan, bukan prediksi baru.')


if selection == '📊 Ringkasan':
    hero('RINGKASAN / OUTPUT GOOGLE COLAB', 'Penelitian Klasifikasi Kegagalan Wafer',
         'Dashboard menyajikan hasil eksperimen yang telah dieksekusi di Google Colab. Setiap metrik dilabeli menurut tahap evaluasinya.')
    a,b,c,d=st.columns(4)
    with a:stat('Wafer', '1.567', 'Train 1.175 · Test 392')
    with b:stat('Sensor awal', '590', '269 fitur akhir setelah cleaning train')
    with c:stat('Fail di seluruh data', '104 (6,64%)', 'Pass 1.463 (93,36%)')
    with d:stat('Model final', 'Extra Trees', 'Dipilih dari PR-AUC CV 5×2')
    left,right=st.columns([.8,1.2],gap='large')
    with left:
        subheading('Distribusi label', 'SECOM dataset · 1.567 wafer')
        show_chart(donut_counts(1463,104))
    with right:
        subheading('Tahap evaluasi tidak boleh tertukar', 'Semua nilai berikut langsung dari notebook')
        phases=pd.DataFrame({
            'Metrik':['PR-AUC','ROC-AUC'],
            'CV 5×2 (mean)':[0.223,0.749],
            'OOF training':[0.186,0.754],
            'Test akhir':[0.240,0.798],
        })
        st.dataframe(phases,hide_index=True,use_container_width=True,
                     column_config={c:st.column_config.NumberColumn(c,format='%.3f') for c in phases.columns if c!='Metrik'})
        panel('Mengapa nilainya tidak sama?',
              'CV digunakan untuk membandingkan 19 algoritma, OOF training untuk memilih threshold, dan test set untuk mengevaluasi model akhir. Berbeda data dan prosedur, sehingga nilainya tidak identik.')
        st.markdown('''<div class="callout"><b>PR-AUC adalah metrik utama seleksi model.</b> Pada kelas Fail yang hanya 6,64%, akurasi tinggi bisa menyesatkan. Semua angka CV dihitung pada 5-fold × 2 pengulangan.</div>''',unsafe_allow_html=True)
    subheading('Hasil test akhir Extra Trees', 'Threshold utama 0,1372 · 392 wafer test, termasuk 26 Fail')
    ci=nb_table('test_ci').rename(columns={'Keterangan':'Metrik','estimasi':'Estimasi','CI95_bawah':'CI 95% bawah',
                                           'CI95_atas':'CI 95% atas','baseline "selalu Pass"':'Baseline selalu Pass'})
    st.dataframe(ci,use_container_width=True,hide_index=True)
    st.caption('CI 95% menggunakan bootstrap 2.000 replikasi, sesuai notebook. Semua nilai adalah proporsi (misalnya 0,798 = 79,8%).')
    a,b,c,d=st.columns(4)
    with a:st.metric('Sensitivitas test','0,808','21 / 26 Fail terdeteksi')
    with b:st.metric('Spesifisitas test','0,708','259 / 366 Pass tepat')
    with c:st.metric('Presisi test','0,164','21 benar dari 128 alarm')
    with d:st.metric('Akurasi test','0,714','NIR test = 0,934')
    st.markdown('''<div class="callout warning"><b>Kesimpulan utama:</b> Extra Trees dapat membantu memprioritaskan inspeksi tambahan, tetapi 107 alarm palsu dan penurunan performa kronologis membuatnya belum layak menjadi keputusan scrap otomatis.</div>''',unsafe_allow_html=True)

elif selection == '🧪 Kualitas & Preprocessing':
    hero('DATA QUALITY / PREPROCESSING','Kualitas Data dan Persiapan Model',
         'Ringkasan kualitas data, perbaikan timestamp, pembagian train/test, dan tahapan preprocessing tanpa data leakage.')
    a,b,c,d=st.columns(4)
    with a:stat('Missing cell','4,5375%','Dari seluruh 590 kolom sensor')
    with b:stat('Sensor missing > 50%','28','Statistik deskriptif seluruh dataset')
    with c:stat('Sensor konstan','116','Hasil pemeriksaan data awal')
    with d:stat('Baris duplikat','0','Sesuai output kualitas data')
    display_nb_table('quality',title='Kualitas data asli (Colab Bagian 3)',height=290)
    st.markdown('''<div class="callout"><b>Perbaikan waktu:</b> 604 baris tanggal dikoreksi. Setelah perbaikan, rentang menjadi 19 Juli – 17 Oktober 2008 dan tidak ada lompatan waktu mundur.</div>''',unsafe_allow_html=True)
    display_nb_table('split',title='Pembagian train dan test',help_text='Stratified 75% : 25% · dari Colab Bagian 4',height=150)
    subheading('Pembersihan fitur saat training','Tahapan hasil DataCleaner pada training set — jangan disamakan dengan statistik data awal')
    stages=pd.DataFrame({
        'Tahap':['Awal','Setelah missing > 50%','Setelah konstan','Setelah korelasi |r| > 0,95'],
        'Fitur tersisa':[590,566,450,269],
    })
    fig=go.Figure(go.Bar(x=stages['Tahap'],y=stages['Fitur tersisa'],marker_color=[BLUE,'#6795D8','#5FA7C7','#25A4AC'],
                         text=stages['Fitur tersisa'],textposition='outside'))
    fig.update_layout(**base_chart(340),yaxis=dict(range=[0,670],title='Fitur tersisa',gridcolor='#E5EDF6'))
    show_chart(fig)
    st.caption('Training: 590 awal − 24 fitur terlalu kosong − 116 konstan − 181 berkorelasi tinggi = 269. Keputusan preprocessing di-fit dalam fold training, bukan seluruh data.')
    st.markdown('''<div class="callout"><b>EDA pendukung:</b> Uji missingness dilakukan pada 103 fitur; 8 memiliki p &lt; 0,05, tetapi <b>0</b> yang signifikan setelah FDR. Analisis korelasi training: 450 fitur, 612 pasangan |r| &gt; 0,80; 280 pasangan |r| &gt; 0,95; 125 pasangan |r| &gt; 0,99. PCA membutuhkan 85 / 126 / 159 komponen untuk menjelaskan 80% / 90% / 95% varians.</div>''',unsafe_allow_html=True)
    if df_upload is not None:
        st.divider()
        subheading('EDA interaktif dari CSV yang diunggah','Bagian ini dihitung ulang dari dataset upload, bukan dari notebook benchmark')
        try:
            sensors=sensor_numeric(df_upload)
            if not sensors.empty:
                a,b,c=st.columns(3)
                a.metric('Baris CSV',len(df_upload))
                b.metric('Fitur numerik',sensors.shape[1])
                c.metric('Missing cell CSV',f'{sensors.isna().mean().mean()*100:.2f}%')
                missing=(sensors.isna().mean()*100).sort_values(ascending=False).head(18)
                fig=go.Figure(go.Bar(x=missing.index.astype(str),y=missing.values,marker_color=BLUE))
                fig.update_layout(**base_chart(320),yaxis_title='Missing (%)',xaxis_title='Sensor')
                show_chart(fig)
                with st.expander('Pratinjau 20 baris dataset yang diunggah'):
                    st.dataframe(df_upload.iloc[:20,:min(15,len(df_upload.columns))],use_container_width=True,hide_index=True)
        except Exception as exc:st.error(f'EDA CSV gagal: {exc}')

elif selection == '🏆 Perbandingan Model':
    hero('MODEL SCREENING / REPEATED STRATIFIED CV','Perbandingan 19 Algoritma',
         'Skor cross-validation Google Colab (5 fold × 2 pengulangan), bukan skor pengujian akhir test set.')
    a,b,c=st.columns(3)
    with a:stat('Model terpilih','Extra Trees','PR-AUC CV = 0,223 ± 0,083')
    with b:stat('ROC-AUC CV Extra Trees','0,749 ± 0,031','Berbeda dengan test ROC-AUC 0,798')
    with c:stat('Baseline acak PR-AUC','0,066','Prevalensi Fail ≈ 0,0664')
    df=df_models()
    metrics={'PR-AUC':'pr_auc','ROC-AUC':'roc_auc','G-mean':'g_mean',
             'Recall / Sensitivitas':'recall','Spesifisitas':'specificity',
             'Balanced accuracy':'bal_acc','MCC':'mcc','Akurasi':'accuracy'}
    l,r=st.columns([2,1])
    with l:metric=st.selectbox('Pilih metrik cross-validation',list(metrics))
    with r:n=st.slider('Jumlah model pada grafik',5,19,12)
    key=metrics[metric]
    vals=df[['Model',key]].copy()
    vals[metric]=vals[key].map(cv_mean)
    vals=vals.sort_values(metric,ascending=False).head(n)
    subheading(f'Peringkat berdasarkan {metric}','Nilai pada grafik adalah rata-rata CV, bukan skor test set')
    plot_horizontal(vals,'Model',metric,'Extra Trees')
    st.markdown('''<div class="callout"><b>Perbandingan valid:</b> 0,749 merupakan rata-rata ROC-AUC Extra Trees saat screening. Nilai 0,798 berasal dari test set setelah model terpilih. G-mean CV Extra Trees = 0 pada threshold default 0,5, bukan pada threshold hasil optimasi.</div>''',unsafe_allow_html=True)
    display_nb_table('screening',title='Tabel lengkap Google Colab — 19 model + 2 baseline',
                     help_text='Semua mean ± SD: PR-AUC, ROC-AUC, G-mean, balanced accuracy, MCC, recall, spesifisitas, akurasi, waktu',height=600)
    st.caption('Untuk membaca grafik, ingat: PR-AUC adalah kriteria seleksi model, bukan akurasi tertinggi. Skor klasifikasi threshold-dependent pada screening memakai threshold default.')

elif selection == '⚙️ Resampling & Tuning':
    hero('EXPERIMENT COMPARISON / GOOGLE COLAB','Resampling, Seleksi Fitur & Tuning',
         'Eksperimen lanjutan mengikuti notebook. Nilai Optuna menggunakan 5-fold dengan seed berbeda dari perbandingan akhir 5×2 CV.')
    tab1,tab2,tab3=st.tabs(['Penanganan kelas','Seleksi fitur','Tuning, ensemble & uji statistik'])
    with tab1:
        subheading('Penanganan ketidakseimbangan kelas','Dua learner: Logistic Regression dan Extra Trees · Colab Bagian 9')
        st.markdown('Resampling dapat mengubah recall/G-mean pada threshold default, tetapi tidak otomatis menaikkan PR-AUC.')
        display_nb_table('resampling',height=400)
    with tab2:
        subheading('Seleksi fitur dan reduksi dimensi','ANOVA, MI, PCA, L1 Logistic, RF importance, dan seluruh fitur · Colab Bagian 10')
        display_nb_table('feature_selection',height=390)
        st.info('Extra Trees tanpa seleksi fitur tambahan tetap mencapai PR-AUC CV 0,223; pemilihan fitur agresif tidak meningkatkan hasil secara konsisten.')
    with tab3:
        a,b=st.columns(2)
        with a:stat('Optuna Extra Trees','0,2110','5-fold tuning, 25 trial (2 pruned)')
        with b:stat('Optuna LightGBM','0,2228','5-fold tuning, 30 trial (6 pruned)')
        st.caption('Nilai Optuna di atas bukan hasil re-evaluasi 5×2 CV. Re-evaluasi: Extra Trees (tuned) 0,217 ± 0,076; LightGBM (tuned) 0,207 ± 0,071.')
        st.markdown('**Ensemble pada 5×2 CV:** Soft voting 0,208 ± 0,060; stacking 0,213 ± 0,067 (PR-AUC).')
        display_nb_table('top_models',title='10 model teratas, termasuk tuning dan ensemble',height=430)
        subheading('Uji perbandingan statistik','Friedman χ²=100,86; df=22; p=4,56×10⁻¹²')
        display_nb_table('significance',height=370)
        st.info('Corrected resampled t-test: delapan pesaing terdekat Extra Trees memiliki p > 0,4. Data belum cukup untuk menyimpulkan perbedaan yang signifikan pada pasangan tersebut.')

elif selection == '🎯 Evaluasi & Threshold':
    hero('OOF TRAINING → HOLDOUT TEST','Evaluasi Model & Pemilihan Threshold',
         'Titik operasi ditetapkan dari prediksi out-of-fold pada training set. Nilai di test set dihitung setelah titik operasi ditetapkan.')
    ops=df_operating()
    selected=st.selectbox('Pilih titik operasi test set',ops['Strategi'].tolist(),index=0)
    row=ops[ops['Strategi']==selected].iloc[0]
    a,b,c,d=st.columns(4)
    with a:stat('Threshold test (dibulatkan)',f"{row['Threshold']:.3f}",'Utama sebenarnya = 0,1372')
    with b:stat('Sensitivitas test',f"{row['Sensitivitas']:.1%}",f"{int(row['TP'])} dari 26 Fail")
    with c:stat('Spesifisitas test',f"{row['Spesifisitas']:.1%}",f"{int(row['TN'])} dari 366 Pass")
    with d:stat('Presisi test',f"{row['Presisi']:.1%}",f"{int(row['TP'])} dari {int(row['TP']+row['FP'])} alarm")
    left,right=st.columns([1,1],gap='large')
    with left:
        subheading('Confusion matrix – TEST','392 wafer, threshold terpilih')
        show_chart(confusion_chart(row))
    with right:
        subheading('Trade-off – TEST','Sensitivitas, spesifisitas, dan presisi')
        fig=go.Figure(go.Bar(x=['Sensitivitas','Spesifisitas','Presisi'],
                             y=[row['Sensitivitas'],row['Spesifisitas'],row['Presisi']],
                             marker_color=[CORAL,BLUE,CYAN],
                             text=[f"{row[c]:.1%}" for c in ['Sensitivitas','Spesifisitas','Presisi']],textposition='outside'))
        fig.update_layout(**base_chart(320),yaxis=dict(range=[0,1.10],tickformat='.0%',gridcolor='#E5EDF6'))
        show_chart(fig)
        panel('Makna confusion matrix','TP = Fail terdeteksi, FN = Fail terlewat, FP = Pass terkena alarm palsu, TN = Pass dikenali benar.')
    display_nb_table('threshold_oof_train',title='1. Hasil OOF TRAINING (untuk memilih threshold)',
                     help_text='Bukan angka test; OOF utama PR-AUC 0,186 dan ROC-AUC 0,754',height=330)
    display_nb_table('threshold_test',title='2. Hasil TEST SET (setelah threshold dipilih)',
                     help_text='Seluruh TP, FN, FP, TN dan 9 metrik dari notebook Colab Bagian 15',height=340)
    display_nb_table('test_ci',title='3. Seluruh metrik TEST dan CI 95%',
                     help_text='Bootstrap 2.000 replikasi; tidak semua metrik mempunyai CI sempit',height=425)
    st.markdown('''<div class="callout warning"><b>Hasil penting:</b> pada threshold default 0,50 semua 26 Fail terlewat meskipun akurasi 0,934. Pada threshold utama G-mean dari OOF (0,1372), 21 Fail terdeteksi, 5 terlewat, dan 107 Pass salah diberi alarm.</div>''',unsafe_allow_html=True)

elif selection == '📈 Drift & Biaya':
    hero('CHRONOLOGICAL VALIDATION / COST','Drift Temporal & Analisis Biaya',
         'Perbandingan acak versus kronologis dan hasil optimasi biaya yang benar-benar dilaporkan Google Colab.')
    left,right=st.columns([1,1],gap='large')
    with left:
        subheading('Fail rate per bulan','Juli–Oktober 2008 · hasil EDA notebook')
        months=pd.DataFrame(MONTHLY_FAIL,columns=['Bulan','Fail rate'])
        fig=go.Figure(go.Bar(x=months['Bulan'],y=months['Fail rate'],
                             text=[f'{x:.1%}' for x in months['Fail rate']],textposition='outside',
                             marker_color=[CORAL,'#ECA1A7','#84AEDF','#3D77D2']))
        fig.update_layout(**base_chart(330),yaxis=dict(range=[0,.27],tickformat='.0%',gridcolor='#E5EDF6'))
        show_chart(fig)
        st.caption('Uji χ² homogenitas antarbulan: 44,11; df=3; p < 0,001 (sesuai notebook).')
    with right:
        subheading('Validasi acak vs kronologis','Angka pengujian dari Google Colab Bagian 17')
        chron=nb_table('chronological')
        chosen=chron[chron['Keterangan'].isin(['pr_auc','roc_auc','sensitivity','g_mean'])]
        fig=go.Figure()
        for col,color in [('Split acak stratified',BLUE),('Split kronologis',CORAL)]:
            fig.add_bar(name=col,x=chosen['Keterangan'],y=chosen[col],marker_color=color,
                        text=[f'{v:.3f}' for v in chosen[col]],textposition='outside')
        fig.update_layout(**base_chart(330),barmode='group',yaxis=dict(range=[0,1],gridcolor='#E5EDF6'),
                          legend=dict(orientation='h',y=-.2))
        show_chart(fig)
        st.caption('Kronologis: training 1.175 wafer lebih awal (80 Fail), test 392 wafer terakhir (24 Fail).')
    display_nb_table('chronological',title='Tabel validasi acak vs kronologis',height=365)
    st.markdown('''<div class="callout warning"><b>Drift:</b> ROC-AUC test turun dari 0,798 menjadi 0,570 dan PR-AUC dari 0,240 menjadi 0,074. Pengujian acak kemungkinan optimistis untuk produksi masa depan.</div>''',unsafe_allow_html=True)
    st.divider()
    subheading('Biaya keputusan dari notebook','Setiap rasio biaya memilih threshold optimumnya sendiri pada OOF TRAINING; hasilnya kemudian diuji pada TEST')
    costs=nb_table('cost')
    display_nb_table('cost',title='Biaya per wafer menurut Google Colab (4 rasio)',height=260)
    picked=st.selectbox('Pilih rasio biaya yang diuji Colab',costs['Keterangan'].astype(str).tolist(),index=1)
    r=costs[costs['Keterangan']==picked].iloc[0]
    comp=pd.DataFrame({'Kebijakan':['Model (threshold OOF terpilih)','Selalu Pass','Periksa semua'],
                       'Biaya per wafer':[r['biaya/wafer model (test)'],r['biaya/wafer "selalu Pass"'],r['biaya/wafer "periksa semua"']]})
    fig=go.Figure(go.Bar(x=comp['Kebijakan'],y=comp['Biaya per wafer'],marker_color=[BLUE,'#9CB5D7',CORAL],
                         text=[f'{v:.3f}' for v in comp['Biaya per wafer']],textposition='outside'))
    fig.update_layout(**base_chart(320),yaxis=dict(range=[0,max(comp['Biaya per wafer'])*1.2],gridcolor='#E5EDF6'))
    show_chart(fig)
    if float(r['penghematan vs strategi terbaik tanpa model'])<0:
        st.warning(f"Pada rasio {picked}, model tidak memberikan penghematan dibanding opsi terbaik tanpa model (nilai notebook {float(r['penghematan vs strategi terbaik tanpa model']):.3f}).")
    else:
        st.success(f"Pada rasio {picked}, penghematan dibanding strategi terbaik tanpa model = {float(r['penghematan vs strategi terbaik tanpa model']):.1%} menurut notebook.")
    st.caption('Tidak ada interpolasi threshold untuk rasio lain; dashboard hanya menampilkan 4 rasio yang benar-benar diuji di Colab. Semua nominal adalah biaya relatif, bukan rupiah.')

elif selection == '🔬 Sensor & Anomali':
    hero('UNIVARIATE / SHAP / ANOMALY DETECTION','Insight Sensor dan Pendekatan Anomali',
         'Semua hasil sensor dan pendekatan unsupervised berasal dari output notebook, dengan metode sumbernya dibedakan.')
    a,b,c=st.columns(3)
    with a:stat('Sensor diuji MWU','474','74 p < 0,05')
    with b:stat('Signifikan setelah FDR','13','Ambang q-FDR < 0,05')
    with c:stat('AUC fitur tunggal max','0,683','Tidak sama dengan ROC-AUC model')
    univ=nb_table('sensor_univariate')
    vals=univ[['Keterangan','AUC_tunggal']].copy().sort_values('AUC_tunggal')
    plot_horizontal(vals,'Keterangan','AUC_tunggal',height=435)
    display_nb_table('sensor_univariate',title='Hasil uji univariat Mann–Whitney U (Colab Bagian 5.3)',height=370)
    st.markdown('''<div class="callout"><b>Penting:</b> SHAP dan permutation importance di Colab dihitung memakai <b>LightGBM tuned</b> untuk interpretasi, bukan Extra Trees final. Grafik AUC sensor tunggal juga bukan feature importance model final.</div>''',unsafe_allow_html=True)
    shap=nb_table('shap')
    l,r=st.columns([1,1],gap='large')
    with l:
        subheading('SHAP (LightGBM tuned)','Mean |SHAP| pada 15 sensor teratas di notebook')
        chart=shap[['Keterangan','mean|SHAP|']].copy()
        plot_horizontal(chart,'Keterangan','mean|SHAP|',height=520)
    with r:
        subheading('Permutation importance (LightGBM tuned)','Penurunan PR-AUC test saat sensor diacak')
        chart=shap[['Keterangan','perm_ΔPR-AUC']].copy().sort_values('perm_ΔPR-AUC')
        fig=go.Figure(go.Bar(x=chart['perm_ΔPR-AUC'],y=chart['Keterangan'],orientation='h',
                             marker_color=[BLUE if v>=0 else CORAL for v in chart['perm_ΔPR-AUC']],
                             text=[f'{v:.3f}' for v in chart['perm_ΔPR-AUC']],textposition='outside'))
        fig.update_layout(**base_chart(520,dict(l=4,r=50,t=10,b=30)),yaxis=dict(automargin=True),
                          xaxis=dict(gridcolor='#E5EDF6'))
        show_chart(fig)
    display_nb_table('shap',title='Tabel SHAP, permutation dan peringkat MWU',height=405)
    st.caption('Sensor konsisten di ≥2 daftar: f103, f129, f130, f205, f21, f213, f247, f33, f587, f59. Karena anonim, perlu metadata manufaktur untuk interpretasi fisik.')
    display_nb_table('anomaly',title='Perbandingan SPC / deteksi anomali tanpa label vs Extra Trees',
                     help_text='Hotelling T², SPE, Isolation Forest, LOF · Colab Bagian 19',height=320)
    st.caption('PCA untuk SPC memakai 96 komponen (80% varians), UCL T² 146,4, UCL SPE 120,1. Pada data ini model supervised memiliki ROC-AUC dan PR-AUC lebih tinggi dari metode anomali.')

else:
    hero('VERIFIKASI / AKADEMIK','Kesimpulan Penelitian & Bukti Angka',
         'Satu tempat untuk memeriksa kembali setiap tabel sumber yang ditampilkan tanpa melatih ulang model.')
    st.markdown('### Kesimpulan berdasarkan Google Colab')
    st.markdown('''
1. **Akurasi bukan metrik pemilihan.** Baseline selalu Pass akurasinya 0,934, tetapi recall Fail 0. Pada syarat akurasi test minimal 90%, model hanya mendeteksi 5 dari 26 Fail.
2. **Model final Extra Trees**, dipilih dari PR-AUC rata-rata CV 5×2 sebesar 0,223 ± 0,083. Perbaikan melalui tuning dan ensemble tidak melampaui hasil tersebut pada CV pembanding.
3. **Test set akhir**: PR-AUC 0,240; ROC-AUC 0,798; sensitivitas 0,808; spesifisitas 0,708; presisi 0,164; G-mean 0,756; MCC 0,273; akurasi 0,714. Ini semua sesuai threshold utama G-mean dari OOF training.
4. **Biaya bergantung rasio FN:FP.** Pada 10:1 penghematan 43,8%; pada 5:1 model belum lebih hemat daripada strategi terbaik tanpa model.
5. **Drift menjadi keterbatasan besar.** Pada split kronologis, PR-AUC 0,074 dan ROC-AUC 0,570; penerapan produksi butuh validasi berurutan dan retraining.
6. **Sensor dan data leakage perlu dibaca dengan tepat.** Hanya 13 dari 474 sensor signifikan setelah FDR; leakage sengaja disimulasikan di notebook untuk menunjukkan nilai terlalu optimistis.
    ''')
    subheading('Validasi data leakage','Eksperimen bocor hanyalah ilustrasi kesalahan metodologi, bukan performa model yang sah')
    display_nb_table('leakage',height=390)
    st.markdown('''<div class="callout warning"><b>Jangan melaporkan PR-AUC = 0,998 sebagai hasil model.</b> Angka itu berasal dari SMOTE/seleksi fitur sebelum split (data leakage). Hasil yang valid adalah PR-AUC test 0,240.</div>''',unsafe_allow_html=True)
    display_nb_table('summary',title='Ringkasan final dari notebook (Colab Bagian 21)',height=350)
    st.divider()
    subheading('Audit tabel output Google Colab','Pilih satu tabel untuk membandingkan angka secara langsung dengan notebook')
    titles={
        'split':'Train/Test (Bagian 4)', 'quality':'Kualitas Data (Bagian 3)',
        'baseline':'Baseline (Bagian 7)', 'screening':'19 Algoritma + Baseline (Bagian 8)',
        'resampling':'Resampling (Bagian 9)', 'feature_selection':'Seleksi Fitur (Bagian 10)',
        'top_models':'Top 10 dan Tuning/Ensemble (Bagian 12)', 'significance':'Signifikansi Model (Bagian 13)',
        'threshold_oof_train':'Threshold dari OOF Training (Bagian 14)',
        'test_ci':'Evaluasi Test + CI95% (Bagian 15)',
        'threshold_test':'Titik Operasi Test (Bagian 15)', 'cost':'Biaya Model (Bagian 16)',
        'chronological':'Validasi Kronologis (Bagian 17)','sensor_univariate':'Uji Sensor (Bagian 5)',
        'shap':'SHAP/Permutation (Bagian 18)','anomaly':'SPC dan Anomali (Bagian 19)',
        'leakage':'Data Leakage (Bagian 20)','summary':'Ringkasan Final (Bagian 21)',
    }
    chosen=st.selectbox('Output notebook yang ingin diperiksa',list(titles),format_func=lambda k:titles[k])
    st.dataframe(nb_table(chosen),use_container_width=True,hide_index=True,height=450)
    st.download_button('Unduh tabel yang dipilih (.csv)',data=nb_table(chosen).to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'google_colab_{chosen}.csv',mime='text/csv')
    st.caption('Tabel asli diekstrak dari output sel notebook, tidak dihitung ulang. Pembulatan 3 desimal pada beberapa tabel mengikuti tampilan Colab. Tanggal/angka dari unggahan CSV dapat berbeda karena data berbeda.')

st.markdown('''<div class="footer-note"><b>SECOM Insight v2.0 · Konsisten dengan output Google Colab.</b> Dataset SECOM 1.567 wafer × 590 sensor. Semua hasil eksperimen merupakan nilai statis dari notebook yang diberikan. Tidak ada training ulang atau prediksi wafer baru pada dashboard ini. Sumber data interaktif CSV, jika diunggah, hanya digunakan untuk EDA terpisah.</div>''',unsafe_allow_html=True)
