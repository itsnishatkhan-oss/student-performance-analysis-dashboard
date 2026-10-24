"""
╔══════════════════════════════════════════════════════════════════════╗
║       Student Performance Analysis Dashboard  —  app.py             ║
║  Tech : Streamlit · Pandas · NumPy · Matplotlib · Seaborn · Sklearn ║
║  Run  : streamlit run app.py                                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# SECTION 0 ▸ Imports
# ──────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ Page Config & Global Style
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens (accent colours reused by charts)
ACCENT_1 = "#6366f1"   # indigo
ACCENT_2 = "#8b5cf6"   # violet
ACCENT_3 = "#06b6d4"   # cyan
ACCENT_4 = "#10b981"   # emerald
ACCENT_5 = "#f59e0b"   # amber
CHART_PALETTE = [ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4, ACCENT_5]

# Inject custom CSS — theme-adaptive (works on light & dark)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --acc1: #6366f1;
    --acc2: #8b5cf6;
    --acc3: #06b6d4;
    --acc4: #10b981;
    --acc5: #f59e0b;
    --card-bg: rgba(128, 128, 128, 0.07);
    --card-border: rgba(128, 128, 128, 0.20);
    --muted: rgba(128, 128, 128, 0.95);
    --radius: 18px;
}

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* ── Subtle ambient background glow ── */
.stApp::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(900px 500px at 8% -5%, rgba(99,102,241,0.16), transparent 60%),
        radial-gradient(800px 480px at 95% 0%, rgba(139,92,246,0.14), transparent 60%),
        radial-gradient(700px 400px at 50% 110%, rgba(6,182,212,0.10), transparent 60%);
}

/* ── Top header & toolbar ── */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="manage-app-button"],
[data-testid="stEnableFullScreenButton"],
.stAppDeployButton, .stDeployButton { display: none !important; }

/* ── Layout spacing ── */
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; max-width: 1280px; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] > div { padding-top: 0.5rem !important; }

/* ── Hero banner ── */
.hero {
    position: relative; overflow: hidden;
    border-radius: 24px; padding: 34px 30px 30px; text-align: center;
    background: linear-gradient(135deg, rgba(99,102,241,0.95), rgba(139,92,246,0.92) 55%, rgba(6,182,212,0.88));
    box-shadow: 0 18px 50px rgba(99,102,241,0.35);
    margin-bottom: 1.2rem;
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%;
    background: rgba(255,255,255,0.10);
}
.hero::before { width: 260px; height: 260px; top: -110px; left: -70px; }
.hero::after  { width: 320px; height: 320px; bottom: -170px; right: -90px; }
.hero h1 {
    position: relative; z-index: 1; margin: 0;
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; color: #fff !important;
}
.hero p { position: relative; z-index: 1; margin: 8px 0 0; color: rgba(255,255,255,0.88); font-size: 0.98rem; }
.hero .chips { position: relative; z-index: 1; margin-top: 16px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.hero .chip {
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.28);
    color: #fff; padding: 4px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    backdrop-filter: blur(6px);
}

/* ── Section headings ── */
.section-title {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.2rem; font-weight: 700; letter-spacing: -0.01em;
    margin: 1.8rem 0 0.9rem;
}
.section-title::before {
    content: ""; width: 5px; height: 1.4rem; border-radius: 4px;
    background: linear-gradient(180deg, var(--acc1), var(--acc2));
}

/* ── Metric cards ── */
[data-testid="stMetric"], [data-testid="metric-container"] {
    position: relative; overflow: hidden;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 20px 22px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[data-testid="stMetric"]::before, [data-testid="metric-container"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--acc1), var(--acc2), var(--acc3));
}
[data-testid="stMetric"]:hover, [data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    border-color: rgba(99,102,241,0.55);
    box-shadow: 0 14px 34px rgba(99,102,241,0.22);
}
[data-testid="stMetricLabel"] p { font-size: 0.8rem !important; font-weight: 600; opacity: 0.75; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, var(--acc1), var(--acc2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Bordered containers (chart cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius) !important;
    border-color: var(--card-border) !important;
    background: var(--card-bg);
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    transition: box-shadow .2s ease, border-color .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(99,102,241,0.40) !important;
    box-shadow: 0 12px 30px rgba(99,102,241,0.14);
}
.chart-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 2px; }
.chart-sub   { font-size: 0.78rem; opacity: 0.6; margin-bottom: 6px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border);
    background: linear-gradient(180deg, rgba(99,102,241,0.08), rgba(139,92,246,0.04) 40%, transparent);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-weight: 700; letter-spacing: -0.01em; }
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(99,102,241,0.45) !important;
    border-radius: 14px !important;
    background: rgba(99,102,241,0.06) !important;
    transition: background .2s, border-color .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(99,102,241,0.12) !important; border-color: var(--acc1) !important;
}
[data-baseweb="tag"] {
    background: linear-gradient(135deg, var(--acc1), var(--acc2)) !important;
    border-radius: 999px !important;
}
[data-baseweb="tag"] span { color: #fff !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    background: var(--card-bg);
    overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden;
    border: 1px solid var(--card-border);
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}

/* ── Buttons ── */
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, var(--acc1), var(--acc2));
    color: #fff; border: none; border-radius: 12px;
    padding: 0.65rem 2rem; font-weight: 700; font-size: 1rem;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 22px rgba(99,102,241,0.35);
    transition: transform .15s, box-shadow .15s, filter .15s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-2px); filter: brightness(1.08);
    box-shadow: 0 12px 30px rgba(139,92,246,0.50); color: #fff;
}
div.stButton > button:active, div.stDownloadButton > button:active { transform: translateY(0); }

/* ── Sliders ── */
[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, var(--acc1), var(--acc2)) !important;
    box-shadow: 0 0 0 5px rgba(99,102,241,0.22) !important;
}

/* ── Stat cards (cleaning report) ── */
.stat-card {
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: var(--radius); padding: 18px 22px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--acc5), #ef4444);
}
.stat-label { font-size: 0.8rem; font-weight: 600; opacity: 0.75; letter-spacing: 0.02em; }
.stat-value { font-size: 1.9rem; font-weight: 800; color: var(--acc5); margin-top: 2px; }

/* ── Legend pills ── */
.legend-row { margin: 14px 0 6px; font-weight: 600; font-size: 0.9rem; }
.pill {
    display: inline-block; padding: 2px 11px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 700; margin-left: 6px;
}
.pill-red    { background: rgba(239,68,68,0.18);  color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
.pill-yellow { background: rgba(245,158,11,0.18); color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
.pill-green  { background: rgba(16,185,129,0.18); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }

/* ── Prediction result box ── */
.pred-box {
    position: relative; overflow: hidden;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 22px; padding: 26px 28px;
    text-align: center; margin-top: 0.4rem;
    box-shadow: 0 14px 40px rgba(99,102,241,0.18);
}
.pred-box::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 5px;
    background: linear-gradient(90deg, var(--acc1), var(--acc2), var(--acc3));
}
.pred-eyebrow {
    font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase;
    opacity: 0.65; font-weight: 700; margin-bottom: 6px;
}
.pred-grade { font-size: 2.1rem; font-weight: 800; margin-bottom: 6px; }
.pred-value { font-size: 3.4rem; font-weight: 800; line-height: 1.05; }
.pred-label { font-size: 0.9rem; opacity: 0.7; margin-top: 6px; }
.pred-meta  { font-size: 0.85rem; opacity: 0.85; }
.pred-divider { height: 1px; background: var(--card-border); margin: 16px 0; border: none; }

/* ── Dividers ── */
hr { border-color: var(--card-border) !important; }

/* ── Footer ── */
.footer {
    text-align: center; font-size: 0.82rem; opacity: 0.7; padding: 1.2rem 0 0.4rem;
}
.footer b { background: linear-gradient(90deg, var(--acc1), var(--acc2));
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ Helper — Matplotlib adaptive theme wrapper
# ──────────────────────────────────────────────────────────────────────
def adaptive_fig(figsize=(8, 4)):
    """Return a Matplotlib figure styled adaptively for both Light & Dark UI."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)  # Transparent figure background
    ax.set_facecolor("none")  # Transparent axes background

    # Neutral gray that reads well on both modes
    gray_color = "#8b92a5"
    ax.tick_params(colors=gray_color, length=0)
    ax.xaxis.label.set_color(gray_color)
    ax.yaxis.label.set_color(gray_color)
    ax.title.set_color(gray_color)

    # Minimal spines + soft horizontal grid
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor((0.5, 0.5, 0.5, 0.25))
    ax.grid(axis="y", color=(0.5, 0.5, 0.5, 0.18), linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ Dataset Generator (used when no CSV is uploaded)
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n: int = 200) -> pd.DataFrame:
    """
    Generate a synthetic student dataset with realistic correlations.
    n = number of student records (default 200).
    """
    np.random.seed(42)
    subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"]
    genders  = ["Male", "Female"]

    student_id   = [f"STU{str(i+1).zfill(4)}" for i in range(n)]
    gender       = np.random.choice(genders, n, p=[0.52, 0.48])
    subject      = np.random.choice(subjects, n)
    study_hours  = np.random.normal(5.5, 2.0, n).clip(0.5, 12.0).round(1)
    attendance   = np.random.normal(78, 12, n).clip(40, 100).round(1)

    # Marks are positively correlated with study hours & attendance + noise
    marks = (
        study_hours * 5.2 +
        attendance  * 0.25 +
        np.random.normal(0, 7, n)
    ).clip(20, 100).round(1)

    df = pd.DataFrame({
        "Student_ID"  : student_id,
        "Gender"      : gender,
        "Subject"     : subject,
        "Study_Hours" : study_hours,
        "Attendance"  : attendance,
        "Marks"       : marks,
    })

    # Inject ~5 % missing values and ~3 % duplicates for realism
    mask = np.random.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[mask, "Marks"] = np.nan

    dup_rows = df.sample(int(0.03 * n), random_state=7)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ──────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ Data Cleaning
# ──────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Clean the raw dataset:
      1. Remove duplicate rows
      2. Fill missing Marks/numerical columns with column median
      3. Fill missing categorical values with 'Unknown'
    Returns cleaned DataFrame, a dict of cleaning stats, and a missing mask.
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure key numeric columns are properly typed (handling string numbers/errors in uploads)
    for col in ["Study_Hours", "Attendance", "Marks"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    raw_shape = df.shape
    dup_count = df.duplicated().sum()

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Capture missing mask and counts after conversions but before filling
    missing_mask = df.isnull()
    missing_count = missing_mask.sum().sum()

    # Ensure key categorical/string columns are filled and cast to string to avoid comparison/sorting errors
    for col in ["Student_ID", "Gender", "Subject"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Fill numeric nulls with their respective median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = df[col].median()
        # Fallback if median is NaN (e.g. if the entire column is empty/NaN)
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    # Catch-all fallback for any other columns
    df = df.fillna("Unknown")

    stats = {
        "raw_rows"     : raw_shape[0],
        "clean_rows"   : df.shape[0],
        "dup_removed"  : int(dup_count),
        "missing_filled": int(missing_count),
    }
    return df, stats, missing_mask


# ──────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ Machine Learning — Linear Regression
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Linear Regression model to predict Marks
    using Study_Hours and Attendance as features.
    Returns: model, scaler (None; no scaling needed for LR here),
             MAE, R² score.
    """
    X = df[["Study_Hours", "Attendance"]]
    y = df["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2  = round(r2_score(y_test, y_pred), 3)

    return model, mae, r2


# ──────────────────────────────────────────────────────────────────────
# Small UI helpers (presentation only)
# ──────────────────────────────────────────────────────────────────────
def chart_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<div class='chart-title'>{title}</div>"
        f"<div class='chart-sub'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ Main App Layout
# ──────────────────────────────────────────────────────────────────────
def main():
    # ── HEADER ────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero'>
        <h1>🎓 Student Performance Dashboard</h1>
        <p>Turn raw student data into insight — analyze, visualize and predict outcomes.</p>
        <div class='chips'>
            <span class='chip'>📊 Analyze</span>
            <span class='chip'>📉 Visualize</span>
            <span class='chip'>🔮 Predict</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.divider()

        # ── Custom dataset upload (Bonus) ─────────────────────────────
        st.markdown("### 📂 Upload Dataset")
        uploaded = st.file_uploader(
            "Upload CSV (optional)", type=["csv"],
            help="Must have columns: Student_ID, Gender, Subject, Study_Hours, Attendance, Marks"
        )
        st.divider()

        # Filters — populated after data is loaded
        st.markdown("### 🔍 Filters")
        gender_options  = st.empty()
        subject_options = st.empty()

    # ── DATA LOADING ───────────────────────────────────────────────────
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            required = {"Student_ID", "Gender", "Subject", "Study_Hours", "Attendance", "Marks"}
            if not required.issubset(set(raw_df.columns)):
                st.error(f"❌ CSV must contain columns: {required}")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.stop()
    else:
        raw_df = generate_dataset(200)

    # ── DATA CLEANING ──────────────────────────────────────────────────
    clean_df, clean_stats, missing_mask = clean_data(raw_df.copy())

    # ── SIDEBAR FILTERS ────────────────────────────────────────────────
    with st.sidebar:
        gender_filter = gender_options.multiselect(
            "Select Gender",
            options=sorted(clean_df["Gender"].unique()),
            default=sorted(clean_df["Gender"].unique()),
        )
        subject_filter = subject_options.multiselect(
            "Select Subject",
            options=sorted(clean_df["Subject"].unique()),
            default=sorted(clean_df["Subject"].unique()),
        )

    # Apply filters
    filtered_df = clean_df[
        clean_df["Gender"].isin(gender_filter) &
        clean_df["Subject"].isin(subject_filter)
    ]

    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters.")
        st.stop()

    # ── DATA CLEANING REPORT ───────────────────────────────────────────
    with st.expander("🧹 Data Cleaning Report", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows",      clean_stats["raw_rows"])
        c2.metric("After Cleaning", clean_stats["clean_rows"])
        c3.markdown(
            f"<div class='stat-card'><div class='stat-label'>Duplicates Removed</div>"
            f"<div class='stat-value'>{clean_stats['dup_removed']}</div></div>",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"<div class='stat-card'><div class='stat-label'>Missing Values Filled</div>"
            f"<div class='stat-value'>{clean_stats['missing_filled']}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='legend-row'>Sample Raw Data "
            "<span class='pill pill-red'>🟥 Missing</span>"
            "<span class='pill pill-yellow'>🟨 Duplicates</span></div>",
            unsafe_allow_html=True,
        )

        # Highlight Raw Data duplicates and missing values
        def style_clean_data(d):
            dup_mask = d.duplicated(keep=False)
            df_style = pd.DataFrame('', index=d.index, columns=d.columns)
            # Yellow for duplicates
            for i in range(len(d)):
                if dup_mask.iloc[i]:
                    df_style.iloc[i, :] = 'background-color: #5c4300; color: #ffcc00'
            # Red for missing
            for col in d.columns:
                null_mask = d[col].isna()
                df_style.loc[null_mask, col] = 'background-color: #5c0000; color: #ff9999'
            return df_style

        # Sort raw_df to bring duplicates to the top so they are easily visible
        raw_df_sorted = raw_df.sort_values(by=list(raw_df.columns), ascending=False)
        styled_raw = raw_df_sorted.style.apply(style_clean_data, axis=None)

        # Output dataframe handling the Streamlit container_width warning
        try:
            st.dataframe(styled_raw, width="stretch", hide_index=True)
        except Exception:
            st.dataframe(styled_raw, use_container_width=True, hide_index=True)

    # ── KPI METRICS ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Total Students",    len(filtered_df))
    m2.metric("📈 Avg Marks",         f"{filtered_df['Marks'].mean():.1f}")
    m3.metric("⏰ Avg Study Hours",   f"{filtered_df['Study_Hours'].mean():.1f} hrs")
    m4.metric("🏫 Avg Attendance",    f"{filtered_df['Attendance'].mean():.1f}%")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 7 ▸ Visualizations
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Visualizations</div>', unsafe_allow_html=True)

    # Row 1: Bar chart + Histogram
    col_left, col_right = st.columns(2)

    # ── 7a. Bar Chart: Average Marks per Subject ─────────────────────
    with col_left:
        with st.container(border=True):
            chart_header("Average Marks per Subject", "Mean marks grouped by subject")
            avg_marks = (
                filtered_df.groupby("Subject")["Marks"]
                .mean().sort_values(ascending=False).reset_index()
            )
            fig, ax = adaptive_fig(figsize=(7, 4))
            bars = ax.bar(
                avg_marks["Subject"], avg_marks["Marks"],
                color=CHART_PALETTE[:len(avg_marks)], edgecolor="none",
                width=0.62, alpha=0.95,
            )
            ax.bar_label(bars, fmt="%.1f", padding=4, color="#8b92a5",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Subject", fontsize=9)
            ax.set_ylabel("Avg Marks", fontsize=9)
            plt.xticks(rotation=20, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── 7b. Histogram: Marks Distribution ───────────────────────────
    with col_right:
        with st.container(border=True):
            chart_header("Marks Distribution", "How student marks are spread")
            fig, ax = adaptive_fig(figsize=(7, 4))
            ax.hist(
                filtered_df["Marks"], bins=20,
                color=ACCENT_1, edgecolor=(1, 1, 1, 0.35), linewidth=0.6, alpha=0.9
            )
            ax.axvline(filtered_df["Marks"].mean(), color=ACCENT_5, linewidth=2,
                       linestyle="--", label=f"Mean: {filtered_df['Marks'].mean():.1f}")
            ax.set_xlabel("Marks", fontsize=9)
            ax.set_ylabel("Number of Students", fontsize=9)

            # Make legend transparent
            legend = ax.legend(fontsize=8, labelcolor="gray")
            legend.get_frame().set_alpha(0.0)
            legend.get_frame().set_edgecolor("none")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Row 2: Scatter + Heatmap
    col_l2, col_r2 = st.columns(2)

    # ── 7c. Scatter: Study Hours vs Marks ────────────────────────────
    with col_l2:
        with st.container(border=True):
            chart_header("Study Hours vs Marks", "Each dot is a student, with trend line")
            fig, ax = adaptive_fig(figsize=(7, 4))
            scatter_colors = {"Male": ACCENT_1, "Female": ACCENT_2}
            for gender, grp in filtered_df.groupby("Gender"):
                ax.scatter(
                    grp["Study_Hours"], grp["Marks"],
                    color=scatter_colors.get(gender, ACCENT_4),
                    alpha=0.70, s=46, label=gender,
                    edgecolors=(1, 1, 1, 0.6), linewidths=0.5,
                )
            # Regression line
            x_vals = np.linspace(
                filtered_df["Study_Hours"].min(),
                filtered_df["Study_Hours"].max(), 100
            )
            m, b = np.polyfit(filtered_df["Study_Hours"], filtered_df["Marks"], 1)
            ax.plot(x_vals, m * x_vals + b, color=ACCENT_5, linewidth=2.2,
                    linestyle="--", label="Trend Line")
            ax.set_xlabel("Study Hours", fontsize=9)
            ax.set_ylabel("Marks", fontsize=9)
            ax.grid(axis="x", color=(0.5, 0.5, 0.5, 0.12), linewidth=0.8)

            # Transparent legend
            legend = ax.legend(fontsize=8, labelcolor="gray")
            legend.get_frame().set_alpha(0.0)
            legend.get_frame().set_edgecolor("none")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── 7d. Heatmap: Correlation Matrix ──────────────────────────────
    with col_r2:
        with st.container(border=True):
            chart_header("Correlation Heatmap", "Relationship strength between numeric features")
            num_cols = ["Study_Hours", "Attendance", "Marks"]
            corr_matrix = filtered_df[num_cols].corr()
            fig, ax = adaptive_fig(figsize=(7, 4))
            ax.grid(False)
            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f", ax=ax,
                cmap=sns.diverging_palette(250, 12, s=85, l=55, as_cmap=True),
                linewidths=2, linecolor=(0.5, 0.5, 0.5, 0.0),
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 11, "weight": "bold"},
            )
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # SECTION 8 ▸ Data Table + Download Button (Bonus)
    # ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">📋 Filtered Dataset '
        '<span class="pill pill-green">🟩 Fixed Missing Values</span></div>',
        unsafe_allow_html=True,
    )

    filtered_missing_mask = missing_mask.loc[filtered_df.index]
    def style_fixed_missing(d):
        df_style = pd.DataFrame('', index=d.index, columns=d.columns)
        for col in d.columns:
            if col in filtered_missing_mask.columns:
                m = filtered_missing_mask[col]
                # Highlight the fixed values in a distinct color (e.g. green)
                df_style.loc[m, col] = 'background-color: #004d22; color: #66ff99'
        return df_style

    styled_filtered = filtered_df.style.apply(style_fixed_missing, axis=None)
    st.dataframe(styled_filtered, use_container_width=True, hide_index=True)

    # Download filtered data as CSV
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_student_data.csv",
        mime="text/csv",
    )

    # ──────────────────────────────────────────────────────────────────
    # SECTION 9 ▸ Machine Learning Model Info
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🤖 Machine Learning — Linear Regression</div>',
                unsafe_allow_html=True)

    model, mae, r2 = train_model(clean_df)

    ml1, ml2, ml3 = st.columns(3)
    ml1.metric("📐 Model",  "Linear Regression")
    ml2.metric("📉 MAE",   f"{mae} marks")
    ml3.metric("📊 R² Score", f"{r2}")

    st.caption(
        "Model trained on full cleaned dataset (not filtered). "
        "Features: **Study Hours** & **Attendance** → Target: **Marks**"
    )

    # ──────────────────────────────────────────────────────────────────
    # SECTION 10 ▸ Prediction UI
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔮 Predict Student Marks</div>',
                unsafe_allow_html=True)

    pred_col, result_col = st.columns([1.2, 1], gap="large")

    with pred_col:
        with st.container(border=True):
            st.markdown("**Enter the student's details below:**")
            study_h = st.slider(
                "📚 Study Hours (per day)", min_value=0.0, max_value=12.0,
                value=5.0, step=0.5,
            )
            attend  = st.slider(
                "🏫 Attendance (%)", min_value=40, max_value=100,
                value=75, step=1,
            )
            predict_btn = st.button("🔮 Predict Marks", use_container_width=True)

    with result_col:
        if predict_btn:
            input_df    = pd.DataFrame([[study_h, attend]],
                                        columns=["Study_Hours", "Attendance"])
            prediction  = model.predict(input_df)[0]
            prediction  = max(0, min(100, round(float(prediction), 1)))

            # Grade label
            if prediction >= 90:
                grade, clr = "A+  🏆", "#22c55e"
            elif prediction >= 75:
                grade, clr = "A   🌟", "#4ade80"
            elif prediction >= 60:
                grade, clr = "B   👍", "#fbbf24"
            elif prediction >= 50:
                grade, clr = "C   📘", "#fb923c"
            else:
                grade, clr = "F   ⚠️", "#f87171"

            st.markdown(f"""
            <div class='pred-box'>
                <div class='pred-eyebrow'>Grade</div>
                <div class='pred-grade' style='color:{clr};'>{grade}</div>
                <hr class='pred-divider'>
                <div class='pred-eyebrow'>Predicted Marks</div>
                <div class='pred-value' style='color:{clr};'>{prediction}</div>
                <hr class='pred-divider'>
                <div class='pred-meta'>
                    📚 Study Hours: <b>{study_h}</b> &nbsp;|&nbsp;
                    🏫 Attendance: <b>{attend}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='pred-box' style='opacity:0.6;'>
                <div class='pred-eyebrow'>Predicted Marks</div>
                <div class='pred-value' style='opacity:0.5;'>——</div>
                <div class='pred-label'>Set inputs and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ──────────────────────────────────────────────────────────────────
    # SECTION 11 ▸ Footer
    # ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
        🎓 <b>Student Performance Dashboard</b> · Built with Streamlit · Foundations of Data Science
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()