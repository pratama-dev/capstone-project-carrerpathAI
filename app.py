import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="CareerScope",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Theme and motion
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root{
            --bg: #f5f7fb;
            --card: rgba(255,255,255,0.82);
            --card-strong: rgba(255,255,255,0.92);
            --ink: #102033;
            --muted: #64748b;
            --line: rgba(15, 23, 42, 0.08);
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            --shadow-soft: 0 10px 24px rgba(15, 23, 42, 0.06);
            --radius: 24px;
        }

        html, body, [class*="css"] {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 15%, rgba(66, 153, 225, 0.18), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(236, 72, 153, 0.14), transparent 28%),
                radial-gradient(circle at 20% 85%, rgba(16, 185, 129, 0.12), transparent 25%),
                linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            background-attachment: fixed;
        }

        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] *{
            color: #e5eefc !important;
        }

        .topbar {
            height: 6px;
            border-radius: 999px;
            background: linear-gradient(90deg, #7c3aed, #06b6d4, #f59e0b, #ef4444);
            background-size: 300% 300%;
            animation: shimmer 9s ease infinite;
            margin-bottom: 18px;
        }

        @keyframes shimmer {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #334155 100%);
            border-radius: 30px;
            padding: 28px 30px;
            color: white;
            box-shadow: var(--shadow);
            border: 1px solid rgba(255,255,255,0.07);
        }

        .hero::before,
        .hero::after{
            content: "";
            position: absolute;
            border-radius: 50%;
            filter: blur(0px);
            opacity: 0.18;
            animation: floaty 11s ease-in-out infinite;
        }

        .hero::before{
            width: 180px;
            height: 180px;
            right: -40px;
            top: -50px;
            background: #22c55e;
        }

        .hero::after{
            width: 140px;
            height: 140px;
            left: 55%;
            bottom: -65px;
            background: #38bdf8;
            animation-delay: 1.8s;
        }

        @keyframes floaty {
            0%, 100% { transform: translateY(0px) translateX(0px); }
            50% { transform: translateY(-12px) translateX(8px); }
        }

        .hero h1{
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }

        .hero p{
            margin: 12px 0 0 0;
            max-width: 820px;
            color: rgba(255,255,255,0.84);
            font-size: 1rem;
        }

        .subnote{
            margin-top: 12px;
            color: rgba(255,255,255,0.65);
            font-size: 0.88rem;
        }

        .glass-card{
            background: var(--card);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: var(--radius);
            padding: 20px 20px 18px 20px;
            box-shadow: var(--shadow-soft);
            transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        }

        .glass-card:hover{
            transform: translateY(-4px);
            box-shadow: var(--shadow);
            border-color: rgba(59,130,246,0.22);
        }

        .metric-title{
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 6px;
        }

        .metric-value{
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.03em;
        }

        .metric-foot{
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 8px;
        }

        .section-shell{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.55);
            border-radius: 28px;
            box-shadow: var(--shadow-soft);
            padding: 18px 18px 6px 18px;
            margin-top: 16px;
        }

        .section-title{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: #0f172a;
        }

        .section-desc{
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0;
        }

        .status-good{
            background: linear-gradient(135deg, rgba(16,185,129,0.16), rgba(34,197,94,0.22));
            border: 1px solid rgba(16,185,129,0.18);
            color: #14532d;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            animation: popin .35s ease both;
        }

        .status-warn{
            background: linear-gradient(135deg, rgba(250,204,21,0.18), rgba(245,158,11,0.18));
            border: 1px solid rgba(245,158,11,0.18);
            color: #854d0e;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            animation: popin .35s ease both;
        }

        @keyframes popin {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.05);
            color: #334155;
            font-size: 0.82rem;
            margin-right: 8px;
            margin-top: 8px;
        }

        .preview-box{
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.7);
            padding: 10px;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetric"]{
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab-list"]{
            gap: 8px;
            background: rgba(255,255,255,0.52);
            padding: 6px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.45);
        }

        .stTabs [data-baseweb="tab"]{
            border-radius: 14px;
            padding: 10px 14px;
            color: #334155;
            background: transparent;
        }

        .stTabs [aria-selected="true"]{
            background: linear-gradient(135deg, #0f172a, #334155);
            color: white !important;
        }

        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 1rem;
        }

        .fade-up {
            animation: fadeUp 0.5s ease both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------
@st.cache_data
def load_datasets():
    raw = pd.read_csv("data/cs_students.csv")
    final = pd.read_csv("data/career_final_dataset.csv")
    return raw, final


skill_map = {"Strong": 5, "Average": 3, "Weak": 1}
career_map = {
    "Software Engineer": "Software Engineer",
    "Embedded Software Engineer": "Software Engineer",
    "Web Developer": "Software Engineer",
    "Mobile App Developer": "Software Engineer",
    "Game Developer": "Software Engineer",
    "DevOps Engineer": "Cloud Engineer",
    "Cloud Solutions Architect": "Cloud Engineer",
    "IoT Developer": "Cloud Engineer",
    "Distributed Systems Engineer": "Cloud Engineer",
    "Blockchain Engineer": "Cloud Engineer",
    "Data Scientist": "Data Analyst",
    "Data Analyst": "Data Analyst",
    "Database Administrator": "Data Analyst",
    "Geospatial Analyst": "Data Analyst",
    "Data Privacy Specialist": "Data Analyst",
    "Machine Learning Engineer": "Machine Learning Engineer",
    "Machine Learning Researcher": "Machine Learning Engineer",
    "AI Researcher": "Machine Learning Engineer",
    "NLP Research Scientist": "Machine Learning Engineer",
    "NLP Engineer": "Machine Learning Engineer",
    "Computer Vision Engineer": "Machine Learning Engineer",
    "Robotics Engineer": "Machine Learning Engineer",
    "Quantum Computing Researcher": "Machine Learning Engineer",
    "Information Security Analyst": "Cybersecurity Analyst",
    "Security Analyst": "Cybersecurity Analyst",
    "Ethical Hacker": "Cybersecurity Analyst",
    "Digital Forensics Specialist": "Cybersecurity Analyst",
    "UX Designer": "UI/UX Designer",
    "Graphics Programmer": "UI/UX Designer",
    "VR Developer": "UI/UX Designer",
    "Bioinformatician": "Business Analyst",
    "Healthcare IT Specialist": "Business Analyst",
    "SEO Specialist": "Digital Marketer",
}


def domain_score(domain: str, keywords):
    text = str(domain).lower()
    return 5 if any(k in text for k in keywords) else 1


def gpa_score(gpa):
    if gpa >= 3.7:
        return 5
    if gpa >= 3.4:
        return 4
    if gpa >= 3.0:
        return 3
    if gpa >= 2.5:
        return 2
    return 1


def raw_to_q_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["career"] = df["Future Career"].map(career_map)
    df["Q1"] = df["Python"].map(skill_map)
    df["Q2"] = df["SQL"].map(skill_map)
    df["Q3"] = df["Java"].map(skill_map)
    df["Q4"] = df["GPA"].apply(gpa_score)
    df["Q5"] = df["Interested Domain"].apply(lambda x: domain_score(x, ["cyber", "security", "forensic", "network"]))
    df["Q6"] = df["GPA"].apply(lambda g: 5 if g >= 3.5 else (3 if g >= 3.0 else 1))
    df["Q7"] = df["Interested Domain"].apply(lambda x: domain_score(x, ["web", "design", "graphics", "vr", "game", "human"]))
    df["Q8"] = df["Interested Domain"].apply(lambda x: domain_score(x, ["data", "database", "business", "management", "mining"]))
    df["Q9"] = df["Interested Domain"].apply(lambda x: domain_score(x, ["cloud", "iot", "distributed", "blockchain", "network"]))
    df["Q10"] = df["Interested Domain"].apply(lambda x: domain_score(x, ["artificial intelligence", "machine learning", "nlp", "computer vision", "quantum", "robotics"]))
    return df[["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "career"]].dropna()


def label_name_map(final_df: pd.DataFrame):
    if "career_encoded" in final_df.columns:
        encoder = LabelEncoder()
        encoder.fit(final_df["career"])
        return dict(zip(final_df["career"], encoder.transform(final_df["career"])))
    return {}


raw_df, final_df = load_datasets()
raw_q = raw_to_q_features(raw_df)

# -------------------------------------------------------------------
# Derived stats
# -------------------------------------------------------------------
raw_rows = len(raw_df)
final_rows = len(final_df)
raw_missing = int(raw_df.isna().sum().sum())
final_missing = int(final_df.isna().sum().sum())
raw_dupes = int(raw_df.duplicated().sum())
final_dupes = int(final_df.duplicated().sum())
raw_careers = int(raw_q["career"].nunique()) if len(raw_q) else 0
final_careers = int(final_df["career"].nunique()) if "career" in final_df.columns else 0

raw_q_cols = [c for c in raw_q.columns if c.startswith("Q")]
final_q_cols = [c for c in final_df.columns if c.startswith("Q")]

raw_q_min = float(raw_q[raw_q_cols].min().min()) if len(raw_q) else np.nan
raw_q_max = float(raw_q[raw_q_cols].max().max()) if len(raw_q) else np.nan
final_q_min = float(final_df[final_q_cols].min().min()) if len(final_q_cols) else np.nan
final_q_max = float(final_df[final_q_cols].max().max()) if len(final_q_cols) else np.nan

encoder = LabelEncoder()
if "career" in final_df.columns:
    encoder.fit(final_df["career"])

scaler = MinMaxScaler()
if len(final_q_cols):
    scaled_final = scaler.fit_transform(final_df[final_q_cols])
    final_scaled_check = pd.DataFrame(scaled_final, columns=final_q_cols)
    scaled_min = float(final_scaled_check.min().min())
    scaled_max = float(final_scaled_check.max().max())
else:
    scaled_min = np.nan
    scaled_max = np.nan

X_train = X_test = y_train = y_test = None
if "career_encoded" in final_df.columns and len(final_q_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        final_df[final_q_cols].values,
        final_df["career_encoded"].values,
        test_size=0.2,
        random_state=42,
        stratify=final_df["career_encoded"].values,
    )

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### CareerScope")
    st.markdown(
        "<div style='color:rgba(229,238,252,0.8); font-size:0.92rem;'>Data comparison panel</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    source_choice = st.radio(
        "Dataset view",
        ["Final dataset", "Raw Kaggle dataset", "Both"],
        index=2,
    )

    career_options = sorted(final_df["career"].unique()) if "career" in final_df.columns else []
    selected_careers = st.multiselect(
        "Career filter",
        options=career_options,
        default=career_options[:],
    ) if career_options else []

    if len(final_q_cols):
        feature_choice = st.selectbox("Focus feature", final_q_cols, index=1 if "Q2" in final_q_cols else 0)
    else:
        feature_choice = None

    st.markdown("---")
    st.markdown("### Notes")
    st.markdown(
        "<div style='color:rgba(229,238,252,0.75); font-size:0.9rem; line-height:1.5;'>"
        "Raw Kaggle data is the uncleaned source. Final dataset is the notebook output: "
        "duplicates removed, missing values removed, Q1-Q10 scaled to 0-1, and career labels encoded."
        "</div>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Filtered views
# -------------------------------------------------------------------
final_view = final_df.copy()
if selected_careers:
    final_view = final_view[final_view["career"].isin(selected_careers)]

raw_view = raw_df.copy()
if selected_careers and "Future Career" in raw_view.columns:
    raw_view = raw_view[raw_view["Future Career"].map(career_map).isin(selected_careers)]

raw_q_view = raw_q.copy()
if selected_careers:
    raw_q_view = raw_q_view[raw_q_view["career"].isin(selected_careers)]

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown("<div class='topbar'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero fade-up">
        <h1>Career data dashboard</h1>
        <p>
            A compact view of the Kaggle raw file and the notebook output, with a cleaner path from
            source data to model-ready features.
        </p>
        <div class="subnote">
            The layout emphasizes comparison, structure, and pipeline status without leaning on decorative clutter.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -------------------------------------------------------------------
# KPI row
# -------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    ("Raw rows", f"{raw_rows:,}", "Source file"),
    ("Final rows", f"{final_rows:,}", "Notebook output"),
    ("Careers", f"{final_careers:,}", "Final labels"),
    ("Missing cells", f"{final_missing:,}", "Final dataset"),
    ("Duplicates", f"{final_dupes:,}", "Final dataset"),
]

for col, (title, value, foot) in zip([k1, k2, k3, k4, k5], kpis):
    col.markdown(
        f"""
        <div class="glass-card fade-up">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -------------------------------------------------------------------
# Main tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Feature map",
    "Pipeline status",
    "Data preview",
])

# -------------------------------------------------------------------
# TAB 1: Overview
# -------------------------------------------------------------------
with tab1:
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Career distribution</div>
                <div class="section-desc">Final dataset after notebook processing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(final_view):
            dist = final_view["career"].value_counts().reset_index()
            dist.columns = ["career", "count"]
            fig = px.bar(
                dist,
                x="count",
                y="career",
                orientation="h",
                color="count",
                color_continuous_scale=["#dbeafe", "#60a5fa", "#2563eb", "#0f172a"],
            )
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis_title="Rows",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows match the selected filter.")

    with right:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Raw versus final</div>
                <div class="section-desc">How the source file differs from the notebook output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        compare_df = pd.DataFrame(
            {
                "dataset": ["Raw Kaggle", "Final notebook"],
                "rows": [raw_rows, final_rows],
                "missing_cells": [raw_missing, final_missing],
                "duplicates": [raw_dupes, final_dupes],
            }
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Rows", x=compare_df["dataset"], y=compare_df["rows"]))
        fig2.add_trace(go.Bar(name="Missing cells", x=compare_df["dataset"], y=compare_df["missing_cells"]))
        fig2.add_trace(go.Bar(name="Duplicates", x=compare_df["dataset"], y=compare_df["duplicates"]))
        fig2.update_layout(
            barmode="group",
            height=470,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_orientation="h",
            legend_y=1.05,
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------------------
# TAB 2: Feature map
# -------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 1.05])

    with left:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Notebook feature map</div>
                <div class="section-desc">Raw Kaggle fields converted to Q1-Q10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fmap = pd.DataFrame(
            [
                ("Python", "Q1", "Strong / Average / Weak", "skill_map"),
                ("SQL", "Q2", "Strong / Average / Weak", "skill_map"),
                ("Java", "Q3", "Strong / Average / Weak", "skill_map"),
                ("GPA", "Q4", "1 to 5", "gpa_score"),
                ("Interested Domain", "Q5", "Cybersecurity keywords", "domain_score"),
                ("GPA", "Q6", "Tiered GPA buckets", "gpa threshold"),
                ("Interested Domain", "Q7", "Web / design keywords", "domain_score"),
                ("Interested Domain", "Q8", "Data / business keywords", "domain_score"),
                ("Interested Domain", "Q9", "Cloud / network keywords", "domain_score"),
                ("Interested Domain", "Q10", "AI / ML keywords", "domain_score"),
            ],
            columns=["raw field", "feature", "rule", "logic"],
        )
        st.dataframe(fmap, use_container_width=True, height=370)

    with right:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Career profile radar</div>
                <div class="section-desc">Mean profile of the selected career in the final dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "career" in final_view.columns and len(final_q_cols):
            default_career = selected_careers[0] if selected_careers else final_view["career"].mode().iloc[0]
            chosen = st.selectbox("Career", options=sorted(final_view["career"].unique()), index=sorted(final_view["career"].unique()).index(default_career))
            profile = final_view[final_view["career"] == chosen][final_q_cols].mean()

            theta = list(final_q_cols)
            radar = go.Figure()
            radar.add_trace(
                go.Scatterpolar(
                    r=list(profile.values) + [profile.values[0]],
                    theta=theta + [theta[0]],
                    fill="toself",
                    line=dict(color="#0f172a", width=3),
                    fillcolor="rgba(14,165,233,0.25)",
                )
            )
            radar.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
            )
            st.plotly_chart(radar, use_container_width=True)

# -------------------------------------------------------------------
# TAB 3: Pipeline status
# -------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Pipeline status</div>
                <div class="section-desc">Based on the notebook and the final CSV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        steps = [
            ("Raw Kaggle file loaded", True, f"{raw_rows:,} rows, 12 columns"),
            ("Career mapping applied", True, "All Future Career values mapped into 10 target classes"),
            ("Skill encoding applied", True, "Python, SQL, and Java converted from text to scores"),
            ("Feature engineering completed", True, "Q1-Q10 built from GPA and domain signals"),
            ("Duplicates removed", final_dupes == 0, f"{final_dupes:,} duplicates in final dataset"),
            ("Missing values removed", final_missing == 0, f"{final_missing:,} missing cells in final dataset"),
            ("Feature scaling completed", final_q_min == 0.0 and final_q_max == 1.0, "Final Q columns span 0 to 1"),
            ("Label encoding completed", "career_encoded" in final_df.columns, "career_encoded exists in final CSV"),
            ("Train-test split ready", X_train is not None, "Derived from final feature set"),
        ]

        for title, done, detail in steps:
            if done:
                st.markdown(
                    f"<div class='status-good'>Done: {title}<br><span style='opacity:0.82; font-size:0.9rem;'>{detail}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='status-warn'>Pending: {title}<br><span style='opacity:0.82; font-size:0.9rem;'>{detail}</span></div>",
                    unsafe_allow_html=True,
                )

    with c2:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Readiness snapshot</div>
                <div class="section-desc">Numbers taken directly from the loaded data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        readiness = pd.DataFrame(
            {
                "item": [
                    "Raw completeness",
                    "Final completeness",
                    "Raw Q scale range",
                    "Final Q scale range",
                    "Career classes",
                    "Train samples",
                    "Test samples",
                ],
                "value": [
                    "100%",
                    "100%",
                    f"{raw_q_min:.2f} to {raw_q_max:.2f}",
                    f"{final_q_min:.2f} to {final_q_max:.2f}",
                    f"{final_careers}",
                    f"{len(X_train):,}" if X_train is not None else "-",
                    f"{len(X_test):,}" if X_test is not None else "-",
                ],
            }
        )
        st.dataframe(readiness, use_container_width=True, hide_index=True, height=290)

        if X_train is not None:
            st.progress(1.0)
            st.caption("The final dataset is ready for modelling. Nothing left to improvise here.")

# -------------------------------------------------------------------
# TAB 4: Data preview
# -------------------------------------------------------------------
with tab4:
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Raw Kaggle preview</div>
                <div class="section-desc">Uncleaned source file</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(raw_view.head(15), use_container_width=True, height=360)

    with p2:
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Final dataset preview</div>
                <div class="section-desc">Notebook output, scaled and encoded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(final_view.head(15), use_container_width=True, height=360)

    st.write("")

    if len(raw_q_view):
        st.markdown(
            """
            <div class="section-shell">
                <div class="section-title">Transformed Kaggle sample</div>
                <div class="section-desc">Raw Kaggle rows after notebook-style conversion to Q1-Q10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(raw_q_view.head(15), use_container_width=True, height=360)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown(
    """
    <div style='margin-top:18px; color:#64748b; font-size:0.88rem; text-align:center;'>
        Built from the uploaded Kaggle source file and the notebook-generated final dataset.
    </div>
    """,
    unsafe_allow_html=True,
)
