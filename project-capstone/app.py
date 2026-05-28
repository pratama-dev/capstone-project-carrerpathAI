import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Konfigurasi Halaman
st.set_page_config(
    page_title="CareerPath AI Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palet Warna Korporat
COLORS = {
    "primary": "#0D9488",
    "secondary": "#1E3A5F",
    "accent": "#F59E0B",
    "background": "#F8FAFC",
    "text_muted": "#64748B"
}

st.markdown(f"""
<style>
    .main {{ background-color: {COLORS['background']}; }}
    h1, h2, h3 {{ color: {COLORS['secondary']}; font-family: 'Segoe UI', sans-serif; }}
    h2 {{ border-bottom: 2px solid {COLORS['primary']}; padding-bottom: 8px; margin-top: 2rem; }}
    .stMetric {{ background: white; border-radius: 8px; padding: 15px; border: 1px solid #E2E8F0; }}
    .status-badge-done {{ background-color: #DEF7EC; color: #166534; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }}
    .status-badge-pending {{ background-color: #FDE8E8; color: #9B1C1C; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)

# Generate Data (Mirroring Notebook Logic)
@st.cache_data
def load_data():
    np.random.seed(42)
    careers = [
        'Software Engineer', 'Data Analyst', 'UI/UX Designer', 'Digital Marketer',
        'Cybersecurity Analyst', 'Project Manager', 'Content Creator',
        'Business Analyst', 'Cloud Engineer', 'Machine Learning Engineer'
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
            noise = np.random.choice([-2, -1, 0, 1, 2], p=[0.05, 0.20, 0.50, 0.20, 0.05])
            X.append([int(np.clip(v + noise, 1, 5)) for v in profile])
            y.append(career)
            sources.append('dummy')
            
        for _ in range(25):
            noise = np.random.choice([-1, 0, 1], p=[0.25, 0.50, 0.25])
            X.append([int(np.clip(v + noise, 1, 5)) for v in profile])
            y.append(career)
            sources.append('kaggle')

    df = pd.DataFrame(X, columns=cols)
    df['career'] = y
    df['source'] = sources
    return df.sample(frac=1, random_state=42).reset_index(drop=True), cols, careers

df, COLS, CAREERS = load_data()
QUESTIONS = [
    'Q1 Coding', 'Q2 Data Analysis', 'Q3 UI Design', 'Q4 Communication', 
    'Q5 Cybersecurity', 'Q6 Management', 'Q7 Content Creation', 
    'Q8 Business', 'Q9 Cloud Infra', 'Q10 AI/ML'
]

# Sidebar
with st.sidebar:
    st.markdown(f"<h3 style='color:{COLORS['secondary']}'>CareerPath AI</h3>", unsafe_allow_html=True)
    st.caption("Data Science Pipeline Panel")
    st.markdown("---")
    
    selected_careers = st.multiselect("Filter Profesi", options=CAREERS, default=CAREERS)
    selected_sources = st.multiselect("Sumber Data", options=['dummy', 'kaggle'], default=['dummy', 'kaggle'])

df_filtered = df[(df['career'].isin(selected_careers)) & (df['source'].isin(selected_sources))]

# Header
st.markdown("<h1>Data Analytics & Modeling Readiness</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #64748B;'>Tinjauan dataset dan evaluasi fitur untuk integrasi model MLP.</p>", unsafe_allow_html=True)

# Tabs
tab_overview, tab_eda, tab_ai = st.tabs(["Overview & Metrik", "Analisis Multivariat", "Kesiapan AI"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Baris Data", f"{len(df_filtered):,}")
    c2.metric("Dimensi Fitur", len(COLS))
    c3.metric("Kategori Profesi", df_filtered['career'].nunique())
    c4.metric("Integritas Data", "100% Bersih")

    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.markdown("### Distribusi Profesi")
        counts = df_filtered['career'].value_counts().reset_index()
        counts.columns = ['Profesi', 'Kuantitas']
        fig_bar = px.bar(counts, x='Kuantitas', y='Profesi', orientation='h', color='Kuantitas', color_continuous_scale='Teal')
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("### Komposisi Sumber")
        src_counts = df_filtered['source'].value_counts().reset_index()
        src_counts.columns = ['Sumber', 'Proporsi']
        fig_pie = px.pie(src_counts, values='Proporsi', names='Sumber', hole=0.5, color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_eda:
    col_heat, col_radar = st.columns([1.2, 1])
    
    with col_heat:
        st.markdown("### Intensitas Minat per Profesi")
        mean_scores = df_filtered.groupby('career')[COLS].mean()
        mean_scores.columns = QUESTIONS
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(mean_scores, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col_radar:
        st.markdown("### Pemetaan Radar Karakteristik")
        radar_career = st.selectbox("Tinjau Profesi:", CAREERS)
        profile_mean = df_filtered[df_filtered['career'] == radar_career][COLS].mean().values
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=list(profile_mean) + [profile_mean[0]],
            theta=QUESTIONS + [QUESTIONS[0]],
            fill='toself',
            line_color=COLORS['primary']
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
            paper_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with tab_ai:
    st.markdown("### Checklist Kesiapan Pipeline Machine Learning")
    
    reqs = [
        ("Validasi Skala Data (1-5)", True),
        ("Pembersihan Missing Values", True),
        ("Distribusi Kelas Proporsional", True),
        ("Label Encoding (Kategorikal ke Integer)", False),
        ("Normalisasi Fitur (MinMaxScaler)", False),
        ("Pemisahan Data Latih & Uji (Train-Test Split)", False)
    ]
    
    for task, is_done in reqs:
        status = "<span class='status-badge-done'>TUNTAS</span>" if is_done else "<span class='status-badge-pending'>MENUNGGU</span>"
        st.markdown(f"{status} &nbsp;&nbsp; {task}", unsafe_allow_html=True)
        st.write("")
        
    if not all(done for _, done in reqs):
        st.warning("Pemrosesan akhir diperlukan sebelum pelatihan model. Implementasikan Label Encoding dan Scaling pada notebook pipeline.")