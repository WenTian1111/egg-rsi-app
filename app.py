import streamlit as st

st.set_page_config(
    page_title="鸡蛋滚落稳定性分析系统",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ═══════ DARK BASE ═══════ */
    .stApp {
        background: #0A0D14;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    /* ─── Grid overlay background (subtle, non-blocking) ─── */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background-image:
            linear-gradient(rgba(0, 180, 216, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 180, 216, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
        animation: gridMove 20s linear infinite;
    }
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(40px, 40px); }
    }

    /* ─── Floating glow orbs ─── */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse 600px 400px at 15% 25%, rgba(0, 180, 216, 0.06) 0%, transparent 70%),
            radial-gradient(ellipse 500px 500px at 75% 60%, rgba(124, 58, 237, 0.05) 0%, transparent 70%),
            radial-gradient(ellipse 300px 300px at 90% 15%, rgba(0, 180, 216, 0.04) 0%, transparent 70%);
        animation: orbFloat 12s ease-in-out infinite alternate;
    }
    @keyframes orbFloat {
        0% { opacity: 0.6; transform: scale(1); }
        100% { opacity: 1; transform: scale(1.05); }
    }

    /* ─── Blocking fix: content above background ─── */
    .main > div {
        position: relative;
        z-index: 1;
    }
    section[data-testid="stSidebar"] {
        position: relative;
        z-index: 2;
    }

    /* ─── Scan line (subtle, overlay-not-blocking) ─── */
    @keyframes scanLine {
        0% { top: -2px; opacity: 0; }
        10% { opacity: 0.15; }
        90% { opacity: 0.15; }
        100% { top: 100%; opacity: 0; }
    }
    .scan-line {
        position: fixed;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 216, 0.3), transparent);
        z-index: 9999;
        pointer-events: none;
        animation: scanLine 4s ease-in-out infinite;
    }

    div[data-testid="stSidebarNav"] { display: none; }

    /* ═══════ HEADER = shimmer + glow ═══════ */
    .main-header {
        text-align: center;
        padding: 1.2rem 0 0.8rem;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(135deg, #00B4D8 0%, #48CAE4 20%, #7C3AED 50%, #00B4D8 80%, #48CAE4 100%);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: headerShimmer 6s ease-in-out infinite;
        position: relative;
        z-index: 1;
        filter: drop-shadow(0 0 20px rgba(0, 180, 216, 0.3));
    }
    .main-header::after {
        content: '';
        display: block;
        width: 120px;
        height: 3px;
        margin: 8px auto 0;
        background: linear-gradient(90deg, transparent, #00B4D8, #7C3AED, transparent);
        border-radius: 2px;
        animation: headerShimmer 3s ease-in-out infinite;
    }
    @keyframes headerShimmer {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    /* ─── Section headers with animated underline ─── */
    .section-header {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-left: 3px solid #00B4D8;
        background: linear-gradient(90deg, rgba(0, 180, 216, 0.08), transparent);
    }

    /* ═══════ CARDS = hover glow + entrance animation ═══════ */
    .card {
        background: #12141E;
        border: 1px solid #2A2D3E;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        z-index: 1;
        overflow: hidden;
        animation: cardFadeIn 0.6s ease-out both;
    }
    .card:nth-child(1) { animation-delay: 0.05s; }
    .card:nth-child(2) { animation-delay: 0.1s; }
    .card:nth-child(3) { animation-delay: 0.15s; }
    .card:nth-child(4) { animation-delay: 0.2s; }

    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .card:hover {
        border-color: #00B4D8;
        box-shadow:
            0 0 15px rgba(0, 180, 216, 0.12),
            0 0 30px rgba(0, 180, 216, 0.06),
            inset 0 0 15px rgba(0, 180, 216, 0.03);
        transform: translateY(-2px);
    }

    /* ─── Metric cards with top glow bar ─── */
    .metric-card {
        background: #12141E;
        border: 1px solid #2A2D3E;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
        overflow: hidden;
        animation: cardFadeIn 0.5s ease-out both;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 25%; right: 25%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00B4D8, transparent);
        opacity: 0;
        transition: all 0.3s ease;
    }
    .metric-card:hover::before {
        opacity: 1;
        left: 10%; right: 10%;
    }
    .metric-card:hover {
        border-color: #00B4D8;
        box-shadow: 0 0 12px rgba(0, 180, 216, 0.1);
        transform: translateY(-1px);
    }
    .metric-value {
        color: #00B4D8;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-label {
        color: #8A8FA6;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* ─── Info/warning/error boxes ─── */
    .info-box {
        background: #12141E;
        border-left: 4px solid #00B4D8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #CCC;
        position: relative;
        z-index: 1;
    }
    .warning-box {
        background: #12141E;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #12141E;
        border-left: 4px solid #EF4444;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* ─── Accent bar ─── */
    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, #00B4D8, #7C3AED, #00B4D8);
        background-size: 200% auto;
        border-radius: 2px;
        margin: 1rem 0;
        animation: headerShimmer 3s ease-in-out infinite;
    }

    /* ═══════ TABS ═══════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #12141E;
        padding: 6px 10px;
        border-radius: 12px;
        border: 1px solid #2A2D3E;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8A8FA6;
        font-weight: 500;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 180, 216, 0.1);
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background: #00B4D8 !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(0, 180, 216, 0.3);
    }

    /* ═══════ METRIC WIDGET ═══════ */
    div[data-testid="stMetric"] {
        background: #12141E;
        border: 1px solid #2A2D3E;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #8A8FA6;
        font-size: 0.75rem;
    }
    div[data-testid="stMetric"] span {
        color: #00B4D8;
    }

    /* ═══════ SIDEBAR ═══════ */
    section[data-testid="stSidebar"] {
        background: #0D0F17;
        border-right: 1px solid #1A1D2E;
    }
    section[data-testid="stSidebar"] .sidebar-content {
        padding: 1rem 0.5rem;
    }
    .sidebar-section {
        color: #8A8FA6;
        font-size: 0.8rem;
        margin: 1rem 0;
    }
    .sidebar-value {
        color: #00B4D8;
        font-weight: 600;
    }

    /* ═══════ BUTTONS ═══════ */
    div[data-testid="stButton"] button {
        transition: all 0.3s ease;
        border: 1px solid #00B4D8 !important;
        color: #00B4D8 !important;
        background: transparent !important;
    }
    div[data-testid="stButton"] button:hover {
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.3);
        transform: translateY(-1px);
        background: rgba(0, 180, 216, 0.08) !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #0098b8, #00B4D8) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(0, 180, 216, 0.4), 0 0 50px rgba(0, 180, 216, 0.1);
        transform: translateY(-2px);
    }

    /* ═══════ PROGRESS BAR ═══════ */
    div[data-testid="stProgress"] > div {
        background-color: #2A2D3E;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00B4D8, #7C3AED);
    }

    /* ═══════ SELECTBOX ═══════ */
    div[data-baseweb="select"] > div {
        background: #12141E !important;
        border-color: #2A2D3E !important;
    }

    /* ═══════ FILE UPLOADER ═══════ */
    div[data-testid="stFileUploader"] section {
        border-color: #2A2D3E !important;
        background: #12141E !important;
        border-style: dashed;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00B4D8 !important;
    }

    /* ═══════ DATA TABLE ═══════ */
    div[data-testid="stTable"] table {
        background: #12141E !important;
    }
    div[data-testid="stTable"] th {
        background: #1A1D2E !important;
        color: #00B4D8 !important;
        font-weight: 600;
    }
    div[data-testid="stTable"] td {
        color: #CCC !important;
        border-color: #2A2D3E !important;
    }

    /* ═══════ DATA FRAME ═══════ */
    div[data-testid="stDataFrame"] {
        background: #12141E !important;
    }

    /* ═══════ RADIO ═══════ */
    div[data-testid="stRadio"] label {
        color: #CCC !important;
    }
    div[data-testid="stRadio"] div[data-checked="true"] label {
        color: #00B4D8 !important;
    }

    /* ═══════ MULTISELECT ═══════ */
    div[data-baseweb="tag"] {
        background: rgba(0, 180, 216, 0.15) !important;
        color: #00B4D8 !important;
    }

    /* ═══════ FOOTER ═══════ */
    .footer {
        text-align: center;
        color: #4A4D5E;
        font-size: 0.7rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #1A1D2E;
        margin-top: 2rem;
    }

    /* ═══════ PULSE GLOW ═══════ */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px rgba(0, 180, 216, 0.2); }
        50% { box-shadow: 0 0 20px rgba(0, 180, 216, 0.4), 0 0 40px rgba(0, 180, 216, 0.1); }
        100% { box-shadow: 0 0 5px rgba(0, 180, 216, 0.2); }
    }
    .result-glow {
        animation: pulseGlow 2s ease-in-out infinite;
    }

    /* ─── Loading spinner override ─── */
    .stSpinner {
        color: #00B4D8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Floating scan line HTML element (pure visual decoration)
st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🥚 系统导航")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">请在右侧页面中进行操作</div>', unsafe_allow_html=True)

st.markdown('<div class="main-header">🥚 鸡蛋滚落稳定性分析系统</div>', unsafe_allow_html=True)

try:
    from pages.showcase import show_showcase
    from pages.prediction import show_prediction
    has_modules = True
except ImportError as e:
    has_modules = False
    import_error = str(e)

tab1, tab2 = st.tabs(["🔬 数据库展示与分析", "🤖 鸡蛋风险预测"])

with tab1:
    if has_modules:
        show_showcase()
    else:
        st.error(f"无法导入展示模块: {import_error}")
        st.info("请确保 pages/showcase.py 文件存在并包含 show_showcase() 函数")

with tab2:
    if has_modules:
        show_prediction()
    else:
        st.error(f"无法导入预测模块: {import_error}")
        st.info("请确保 pages/prediction.py 文件存在并包含 show_prediction() 函数")



