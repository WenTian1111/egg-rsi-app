"""
鸡蛋滚落稳定性分析系统 - 主入口
西南大学大学生创新创业训练计划项目 (S202510635378)
"""
import os
import streamlit as st

st.set_page_config(
    page_title="鸡蛋滚落稳定性分析系统 | Egg RSI Analyzer",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
# 顶级科技感暗黑工业设计系统 (Cyber-Sci Deep UI System)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg-deep: #07090E;
        --bg-surface: #0E131F;
        --bg-surface-elevated: #141B2D;
        --bg-card: rgba(16, 23, 41, 0.75);
        --bg-card-hover: rgba(22, 32, 56, 0.88);
        --border-subtle: rgba(255, 255, 255, 0.07);
        --border-active: rgba(0, 229, 255, 0.35);
        --cyan-primary: #00E5FF;
        --cyan-glow: rgba(0, 229, 255, 0.25);
        --blue-tech: #0077B6;
        --purple-neon: #8B5CF6;
        --purple-glow: rgba(139, 92, 246, 0.25);
        --green-safe: #10B981;
        --amber-warn: #F59E0B;
        --red-danger: #EF4444;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', Consolas, monospace;
        --font-display: 'Space Grotesk', 'Inter', sans-serif;
    }

    /* ══════ 全局重置与基础背景 ══════ */
    .stApp {
        background-color: var(--bg-deep);
        font-family: var(--font-sans);
        color: var(--text-primary);
        overflow-x: hidden;
    }

    /* 科技感动态环境光背景 */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse 900px 550px at 10% 15%, rgba(0, 229, 255, 0.055) 0%, transparent 70%),
            radial-gradient(ellipse 800px 600px at 85% 65%, rgba(139, 92, 246, 0.05) 0%, transparent 70%),
            radial-gradient(ellipse 600px 450px at 50% 90%, rgba(0, 119, 182, 0.04) 0%, transparent 65%);
    }

    /* 微妙的科技点阵背景 */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        background-image: radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px);
        background-size: 32px 32px;
        opacity: 0.6;
    }

    .main > div {
        position: relative;
        z-index: 1;
        padding-top: 1rem;
    }

    /* 隐藏多余默认组件 */
    div[data-testid="stSidebarNav"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    button[kind="headerNoPadding"] { display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ══════ 自定义科技滚动条 ══════ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-deep);
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--cyan-primary);
    }

    /* ══════ 顶部系统看板标头 ══════ */
    .app-header-container {
        position: relative;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.8rem;
        background: linear-gradient(135deg, rgba(14, 20, 35, 0.85) 0%, rgba(20, 27, 48, 0.65) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .app-header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyan-primary), var(--purple-neon), transparent);
    }
    .app-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .app-branding {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .app-logo-box {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(0, 229, 255, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
    }
    .app-title-text {
        font-family: var(--font-display);
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #FFFFFF 30%, #A5F3FC 70%, #E0E7FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .app-subtitle-text {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 0.25rem;
        letter-spacing: 0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .app-status-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--border-subtle);
        color: var(--text-secondary);
    }
    .header-badge.highlight {
        background: rgba(0, 229, 255, 0.08);
        border-color: rgba(0, 229, 255, 0.3);
        color: var(--cyan-primary);
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--green-safe);
        box-shadow: 0 0 8px var(--green-safe);
        display: inline-block;
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* ══════ 顶部系统快速信息条 ══════ */
    .system-metrics-ribbon {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.85rem;
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .ribbon-item {
        background: rgba(10, 15, 26, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .ribbon-icon {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    .ribbon-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .ribbon-value {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: var(--font-mono);
    }

    /* ══════ 玻璃拟态卡片 ══════ */
    .sci-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .sci-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(0, 229, 255, 0.3);
        box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.5), 0 0 20px -6px var(--cyan-glow);
        transform: translateY(-2px);
    }
    .sci-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.12), transparent);
    }

    /* ══════ 高级工业视觉相框 ══════ */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: #0B0F19 !important;
        box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.6) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        max-height: 380px !important;
        object-fit: contain !important;
    }
    div[data-testid="stImage"] img:hover {
        transform: translateY(-2px) scale(1.015) !important;
        border-color: rgba(0, 229, 255, 0.45) !important;
        box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.8), 0 0 20px -4px rgba(0, 229, 255, 0.3) !important;
    }
    div[data-testid="stImage"] [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        margin-top: 0.4rem !important;
    }

    /* ══════ 区域标头 ══════ */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 1.2rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .section-title-icon {
        color: var(--cyan-primary);
        font-size: 1.2rem;
    }

    /* ══════ TAB 标签页美化 (Pill Switcher) ══════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(14, 20, 35, 0.9);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        margin-bottom: 1.4rem;
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        transition: all 0.25s ease;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.45) !important;
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.2);
    }

    /* ══════ 侧边栏深度重塑 ══════ */
    section[data-testid="stSidebar"] {
        background-color: #0A0D15 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.5rem 1rem;
    }
    .sidebar-brand-card {
        background: linear-gradient(135deg, rgba(16, 23, 42, 0.8) 0%, rgba(10, 15, 26, 0.9) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .sidebar-brand-title {
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.1rem;
        color: #FFFFFF;
        margin-top: 0.4rem;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
    }

    /* 侧边栏图片优雅化容器 */
    .sidebar-img-card {
        background: rgba(14, 20, 35, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    .sidebar-img-card:hover {
        border-color: rgba(0, 229, 255, 0.35);
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.15);
    }

    /* ══════ 按钮科技感 ══════ */
    div[data-testid="stButton"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 0.55rem 1.2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(16, 23, 42, 0.7) !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--cyan-primary) !important;
        color: #FFFFFF !important;
        background: rgba(0, 229, 255, 0.1) !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.25) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00C8E5 0%, #0096C7 100%) !important;
        box-shadow: 0 6px 25px rgba(0, 229, 255, 0.5) !important;
        transform: translateY(-2px);
    }

    /* ══════ 输入框与选择框 ══════ */
    div[data-baseweb="select"] > div {
        background: #0E131F !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: var(--cyan-primary) !important;
    }

    /* ══════ 文件上传器 ══════ */
    div[data-testid="stFileUploader"] section {
        background: rgba(14, 20, 35, 0.5) !important;
        border: 2px dashed rgba(0, 229, 255, 0.25) !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--cyan-primary) !important;
        background: rgba(0, 229, 255, 0.04) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.15) !important;
    }

    /* ══════ 折叠框 Expander ══════ */
    div[data-testid="stExpander"] {
        background: rgba(14, 20, 35, 0.6) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 229, 255, 0.3) !important;
    }

    /* ══════ 指标微型卡片与统计卡 ══════ */
    .metric-cell {
        background: rgba(11, 16, 29, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        transition: all 0.2s ease;
        position: relative;
    }
    .metric-cell:hover {
        border-color: rgba(0, 229, 255, 0.35);
        background: rgba(16, 24, 42, 0.8);
        transform: translateY(-1px);
    }
    .metric-cell-value {
        font-family: var(--font-mono);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--cyan-primary);
        line-height: 1.3;
    }
    .metric-cell-label {
        font-size: 0.74rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ══════ 响应式调整 ══════ */
    @media (max-width: 768px) {
        .app-title-text { font-size: 1.35rem; }
        .app-header-container { padding: 1.1rem; }
        .system-metrics-ribbon { grid-template-columns: 1fr 1fr; }
        .stTabs [data-baseweb="tab"] { padding: 0.45rem 0.8rem; font-size: 0.82rem; }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# 侧边栏：系统控制中心 & 学术成果导览
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <div style="font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(0,229,255,0.4));">🥚</div>
        <div class="sidebar-brand-title">Egg RSI System</div>
        <div class="sidebar-brand-sub">鸡蛋滚落稳定性智能分析 v2.5 Pro</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ 系统控制台")

    # 学术背景机理卡片（自适应优雅预览，不拉伸）
    physics_img = os.path.join(os.path.dirname(__file__), 'assets', 'egg_physics.png')
    if os.path.exists(physics_img):
        with st.expander("🔬 动力学影响机理图", expanded=False):
            st.image(physics_img, caption="图: 静态形态异向性对滚动稳定性的影响机理", use_container_width=True)
            st.caption("形态异向 ➔ 接触迁移 ➔ 姿态扰动 ➔ 稳定性下降")

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0.4rem; color: #64748B; font-size: 0.75rem; text-align: center;">
        🏛️ <b>西南大学创新训练项目</b><br>
        指导教师：李长营 教授<br>
        项目编号：S202510635378
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# 顶部系统控制台标头 (Interactive Header & Telemetry Ribbon)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header-container">
    <div class="app-title-row">
        <div class="app-branding">
            <div class="app-logo-box">🥚</div>
            <div>
                <div class="app-title-text">鸡蛋滚落稳定性智能分析系统</div>
                <div class="app-subtitle-text">
                    <span>Egg Roll Stability Intelligent Analyzer</span>
                    <span style="opacity: 0.4;">|</span>
                    <span style="color: #38BDF8;">基于计算机视觉与多模态机器学习融合</span>
                </div>
            </div>
        </div>
        <div class="app-status-badges">
            <div class="header-badge highlight">
                <span class="status-pulse"></span>
                <span>推理核心：在线就绪</span>
            </div>
            <div class="header-badge">
                <span>🏛️ 西南大学大创项目</span>
            </div>
            <div class="header-badge">
                <span>编号: S202510635378</span>
            </div>
        </div>
    </div>
    <div class="system-metrics-ribbon">
        <div class="ribbon-item">
            <div class="ribbon-icon">📦</div>
            <div>
                <div class="ribbon-label">标本样本库</div>
                <div class="ribbon-value">90 枚标准化样本</div>
            </div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-icon">📐</div>
            <div>
                <div class="ribbon-label">视觉解耦特征</div>
                <div class="ribbon-value">19 维静态形态参数</div>
            </div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-icon">🏆</div>
            <div>
                <div class="ribbon-label">最优决策模型</div>
                <div class="ribbon-value" style="color: #00E5FF;">SVM (AUC: 85.8%)</div>
            </div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-icon">⚡</div>
            <div>
                <div class="ribbon-label">全流程推理延迟</div>
                <div class="ribbon-value" style="color: #10B981;">&lt; 35 ms / 枚</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# 页面模块路由与主容器
# ═══════════════════════════════════════════════════════════════════════
try:
    from pages.showcase import show_showcase
    from pages.prediction import show_prediction
    has_modules = True
except ImportError as e:
    has_modules = False
    import_error = str(e)

tab1, tab2 = st.tabs([
    "🔬 数据库展示与多维分析 (Showcase & Telemetry)",
    "🤖 鸡蛋风险智能预测与分选 (AI Prediction Studio)"
])

with tab1:
    if has_modules:
        show_showcase()
    else:
        st.error(f"❌ 无法导入展示模块: {import_error}")
        st.info("请确保 pages/showcase.py 存在并可加载")

with tab2:
    if has_modules:
        show_prediction()
    else:
        st.error(f"❌ 无法导入预测模块: {import_error}")
        st.info("请确保 pages/prediction.py 存在并可加载")
