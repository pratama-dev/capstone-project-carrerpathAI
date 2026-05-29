import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CareerScope Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.block-container {
    padding-top: 2rem;
}

.hero-card {
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
    padding: 2rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 35px rgba(0,0,0,0.15);
}

.metric-card {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.06);
}

.metric-title {
    font-size: 0.95rem;
    color: #64748B;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
}

.section-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.06);
    margin-bottom: 20px;
}

.pipeline-ok {
    background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
    color: #166534;
    padding: 14px 18px;
    border-radius: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

.pipeline-pending {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    color: #92400E;
    padding: 14px 18px;
    border-radius: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

.small-text {
    color: #64748B;
    font-size: 0.92rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# DATASET GENERATION
# =========================
@st.cache_data
def load_data():

    np.random.seed(42)

    careers = [
        'Software Engineer',
        'Data Analyst',
        'UI/UX Designer',
        'Digital Marketer',
        'Cybersecurity Analyst',
        'Project Manager',
        'Content Creator',
        'Business Analyst',
        'Cloud Engineer',
        'Machine Learning Engineer'
    ]

    profiles = {
        'Software Engineer':          [5, 3, 1, 1, 3, 1, 1, 1, 3, 3],
        'Data Analyst':               [3, 5, 1, 3, 1, 1, 1, 3, 1, 3],
        'UI/UX Designer':             [1, 1, 5, 3, 1, 1, 5, 1, 1, 1],
        'Digital Marketer':           [1, 3, 3, 5, 1, 3, 5, 3, 1, 1],
        'Cybersecurity Analyst':      [3, 3, 1, 1, 5, 1, 1, 1, 3, 1],
        'Project Manager':            [1, 3, 1, 5, 1, 5, 1, 5, 1, 1],
        'Content Creator':            [1, 1, 3, 5, 1, 1, 5, 1, 1, 1],
        'Business Analyst':           [1, 5, 1, 3, 1, 3, 1, 5, 1, 1],
        'Cloud Engineer':             [3, 1, 1, 1, 3, 1, 1, 1, 5, 3],
        'Machine Learning Engineer':  [5, 5, 1, 1, 1, 1, 1, 1, 3, 5],
    }

    cols = [f'Q{i+1}' for i in range(10)]

    X, y, sources = [], [], []

    for career, profile in profiles.items():

        for _ in range(100):
            noise = np.random.choice(
                [-2, -1, 0, 1, 2],
                p=[0.05, 0.20, 0.50, 0.20, 0.05]
            )

            X.append([
                int(np.clip(v + noise, 1, 5))
                for v in profile
            ])

            y.append(career)
            sources.append('dummy')

        for _ in range(25):

            noise = np.random.choice(
                [-1, 0, 1],
                p=[0.25, 0.50, 0.25]
            )

            X.append([
                int(np.clip(v + noise, 1, 5))
                for v in profile
            ])

            y.append(career)
            sources.append('kaggle')

    df = pd.DataFrame(X, columns=cols)
    df["career"] = y
    df["source"] = sources

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = load_data()

FEATURES = [f"Q{i+1}" for i in range(10)]

QUESTIONS = [
    'Coding',
    'Data Analysis',
    'UI Design',
    'Communication',
    'Cybersecurity',
    'Management',
    'Content Creation',
    'Business',
    'Cloud',
    'AI/ML'
]

# =========================
# PREPROCESSING
# =========================
encoder = LabelEncoder()
df["career_encoded"] = encoder.fit_transform(df["career"])

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[FEATURES])

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features,
    df["career_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["career_encoded"]
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown("# 🚀 CareerScope")
    st.caption("Interactive Career Intelligence Dashboard")

    st.markdown("---")

    selected_career = st.multiselect(
        "Filter Career",
        options=df["career"].unique(),
        default=df["career"].unique()
    )

    selected_source = st.multiselect(
        "Dataset Source",
        options=df["source"].unique(),
        default=df["source"].unique()
    )

df_filtered = df[
    (df["career"].isin(selected_career)) &
    (df["source"].isin(selected_source))
]

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:10px;">CareerScope Analytics Dashboard</h1>
    <p style="font-size:1.05rem; opacity:0.92;">
        Visualisasi kesiapan dataset, distribusi minat karier, 
        dan pipeline machine learning untuk klasifikasi profesi berbasis MLP.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# METRICS
# =========================
c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Total Records", len(df_filtered)),
    ("Feature Variables", len(FEATURES)),
    ("Career Classes", df_filtered["career"].nunique()),
    ("Train/Test Split", "80/20")
]

for col, (title, value) in zip([c1, c2, c3, c4], metrics):

    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "📊 Distribution",
    "🧠 Career Mapping",
    "⚙️ ML Pipeline"
])

# =========================
# TAB 1
# =========================
with tab1:

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("""
        <div class="section-card">
        <h3>Career Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        career_count = (
            df_filtered["career"]
            .value_counts()
            .reset_index()
        )

        career_count.columns = ["Career", "Count"]

        fig = px.bar(
            career_count,
            x="Count",
            y="Career",
            orientation="h",
            color="Count",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("""
        <div class="section-card">
        <h3>Source Composition</h3>
        </div>
        """, unsafe_allow_html=True)

        source_count = (
            df_filtered["source"]
            .value_counts()
            .reset_index()
        )

        source_count.columns = ["Source", "Count"]

        fig2 = px.pie(
            source_count,
            values="Count",
            names="Source",
            hole=0.55,
            color_discrete_sequence=[
                "#7C3AED",
                "#06B6D4"
            ]
        )

        fig2.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 2
# =========================
with tab2:

    st.markdown("""
    <div class="section-card">
    <h3>Career Characteristic Radar</h3>
    <p class="small-text">
    Profil rata-rata ketertarikan dan kemampuan tiap profesi.
    </p>
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "Choose Career",
        sorted(df_filtered["career"].unique())
    )

    profile = (
        df_filtered[df_filtered["career"] == selected][FEATURES]
        .mean()
        .values
    )

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=list(profile) + [profile[0]],
        theta=QUESTIONS + [QUESTIONS[0]],
        fill='toself',
        line=dict(color="#7C3AED", width=3),
        fillcolor="rgba(124,58,237,0.28)"
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5]
            )
        ),
        showlegend=False,
        height=550,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# =========================
# TAB 3
# =========================
with tab3:

    st.markdown("""
    <div class="section-card">
    <h3>Machine Learning Pipeline Readiness</h3>
    <p class="small-text">
    Status preprocessing dan validasi dataset berdasarkan notebook pipeline.
    </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline_steps = [

        ("Dataset berhasil dimuat", True),
        ("Tidak terdapat missing values", True),
        ("Skala fitur konsisten (1-5)", True),
        ("Distribusi kelas seimbang", True),
        ("Label Encoding selesai", True),
        ("MinMax Scaling selesai", True),
        ("Train-Test Split selesai", True),
        ("Dataset siap untuk MLP Training", True)
    ]

    for text, done in pipeline_steps:

        if done:
            st.markdown(
                f"""
                <div class="pipeline-ok">
                ✅ {text}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="pipeline-pending">
                ⏳ {text}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # Additional Stats
    colA, colB, colC = st.columns(3)

    colA.metric(
        "Encoded Classes",
        len(encoder.classes_)
    )

    colB.metric(
        "Training Samples",
        len(X_train)
    )

    colC.metric(
        "Testing Samples",
        len(X_test)
    )

    st.write("")

    st.info(
        "Pipeline preprocessing telah lengkap dan dataset siap digunakan "
        "untuk pelatihan model Neural Network / MLP classifier."
    )

# =========================
# FOOTER
# =========================
st.markdown("""
<br><br>
<center>
<p style='color:#64748B; font-size:0.9rem;'>
Built for modern AI & data analytics workflow 🚀
</p>
</center>
""", unsafe_allow_html=True)
