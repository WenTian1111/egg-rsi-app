import streamlit as st

st.set_page_config(
    page_title="鸡蛋滚落稳定性分析系统",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* === Animated Background === */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
        position: relative;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(0, 180, 216, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(124, 58, 237, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(0, 180, 216, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
        animation: gradientShift 8s ease-in-out infinite alternate;
    }

    @keyframes gradientShift {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }

    /* === Subtle Grid Overlay === */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    div[data-testid="stSidebarNav"] { display: none; }

    .main-header {
        color: #00B4D8;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00B4D8 0%, #7C3AED 50%, #00B4D8 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .section-header {
        color: #FFFFFF;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 0.5rem 0;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #00B4D8, #7C3AED) 1;
        margin: 1.5rem 0 1rem 0;
    }

    .card {
        background-color: #1A1C23;
        border: 1px solid #2D2D3D;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }

    .card:hover {
        border-color: #00B4D8;
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.15), 0 0 30px rgba(0, 180, 216, 0.05);
        transform: translateY(-1px);
    }

    .metric-card {
        background-color: #1A1C23;
        border: 1px solid #2D2D3D;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00B4D8, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover::before {
        opacity: 1;
    }

    .metric-card:hover {
        border-color: #00B4D8;
        box-shadow: 0 0 12px rgba(0, 180, 216, 0.1);
    }

    .metric-value {
        color: #00B4D8;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .metric-label {
        color: #ADB5BD;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, #00B4D8, #7C3AED, #00B4D8);
        background-size: 200% auto;
        border-radius: 2px;
        margin: 1rem 0;
        animation: shimmer 3s ease-in-out infinite;
    }

    .info-box {
        background-color: #1A1C23;
        border-left: 4px solid #00B4D8;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
        position: relative;
        z-index: 1;
    }

    .warning-box {
        background-color: #1A1C23;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
    }

    .error-box {
        background-color: #1A1C23;
        border-left: 4px solid #EF4444;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1A1C23;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #2D2D3D;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #ADB5BD;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 180, 216, 0.1);
        color: #FFFFFF;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00B4D8 !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background-color: #1A1C23;
        border: 1px solid #2D2D3D;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stMetric"] label {
        color: #ADB5BD;
    }

    div[data-testid="stMetric"] span {
        color: #00B4D8;
    }

    .sidebar-content {
        padding: 1rem 0.5rem;
    }

    .sidebar-section {
        color: #ADB5BD;
        font-size: 0.85rem;
        margin: 1rem 0;
    }

    .sidebar-value {
        color: #00B4D8;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.75rem;
        padding: 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid #2D2D3D;
    }

    /* === Pulse Animation for Risk Results === */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px rgba(0, 180, 216, 0.2); }
        50% { box-shadow: 0 0 20px rgba(0, 180, 216, 0.4); }
        100% { box-shadow: 0 0 5px rgba(0, 180, 216, 0.2); }
    }

    .result-glow {
        animation: pulseGlow 2s ease-in-out infinite;
    }

    /* === Scan Line Effect === */
    @keyframes scanLine {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }

    .main-header::after {
        display: none;
    }

    /* === Button Style Enhancement === */
    div[data-testid="stButton"] button {
        transition: all 0.3s ease;
        border: 1px solid #00B4D8 !important;
    }

    div[data-testid="stButton"] button:hover {
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.3);
        transform: translateY(-1px);
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #00B4D8, #0098b8) !important;
        border: none !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 0 20px rgba(0, 180, 216, 0.4);
    }

    /* === Progress Bar Styling === */
    div[data-testid="stProgress"] > div {
        background-color: #2D2D3D;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00B4D8, #7C3AED);
    }

    /* === Selectbox Styling === */
    div[data-baseweb="select"] > div {
        background-color: #1A1C23 !important;
        border-color: #2D2D3D !important;
    }

    /* === File Uploader Styling === */
    div[data-testid="stFileUploader"] section {
        border-color: #2D2D3D !important;
        background-color: #1A1C23 !important;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #00B4D8 !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🥚 系统导航")
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
    st.markdown("请在右侧页面中进行操作")

st.markdown('<div class="main-header">鸡蛋滚落稳定性分析系统</div>', unsafe_allow_html=True)

try:
    from pages.showcase import show_showcase
    from pages.prediction import show_prediction
    has_modules = True
except ImportError as e:
    has_modules = False
    import_error = str(e)

tab1, tab2 = st.tabs(["🔬 展示", "🤖 预测"])

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

st.markdown("""
<div class="footer">
    西南大学 动物科学技术学院
</div>
""", unsafe_allow_html=True)