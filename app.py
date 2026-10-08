"""
鸡蛋滚落稳定性智能分析系统 - 主入口
工业计算机视觉与多模态机器学习驱动
"""
import os
import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

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
        --bg-deep: #050811;
        --bg-surface: #0B1120;
        --bg-surface-elevated: #111A30;
        --bg-card: rgba(15, 23, 42, 0.72);
        --bg-card-hover: rgba(22, 33, 60, 0.88);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-active: rgba(0, 229, 255, 0.45);
        --cyan-primary: #00E5FF;
        --cyan-glow: rgba(0, 229, 255, 0.3);
        --blue-tech: #0077B6;
        --purple-neon: #8B5CF6;
        --purple-glow: rgba(139, 92, 246, 0.3);
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

    /* ══════ 全局重置与动态极光环境光 ══════ */
    .stApp {
        background-color: var(--bg-deep);
        font-family: var(--font-sans);
        color: var(--text-primary);
        overflow-x: hidden;
    }

    /* 动态极光环境微光 */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(ellipse 950px 600px at 12% 12%, rgba(0, 229, 255, 0.07) 0%, transparent 70%),
            radial-gradient(ellipse 850px 650px at 88% 68%, rgba(139, 92, 246, 0.065) 0%, transparent 70%),
            radial-gradient(ellipse 700px 500px at 50% 95%, rgba(0, 119, 182, 0.05) 0%, transparent 65%);
        animation: auroraPulse 12s ease-in-out infinite alternate;
    }
    @keyframes auroraPulse {
        0% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.03); }
        100% { opacity: 0.85; transform: scale(0.98); }
    }

    /* 高精密数字点阵网格 */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 0;
        background-image: 
            radial-gradient(rgba(0, 229, 255, 0.08) 1px, transparent 1px),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 32px 32px, 64px 64px, 64px 64px;
        opacity: 0.65;
    }

    .main > div {
        position: relative;
        z-index: 1;
        padding-top: 0.8rem;
    }

    /* 隐藏 Streamlit 冗余原生部件 */
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
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cyan-primary); }

    /* ══════ 顶部系统控制台超级标头 ══════ */
    .app-header-container {
        position: relative;
        padding: 1.6rem 2rem;
        margin-bottom: 1.6rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(10, 15, 28, 0.92) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        overflow: hidden;
    }
    .app-header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 5%, var(--cyan-primary) 35%, var(--purple-neon) 65%, transparent 95%);
        background-size: 200% 100%;
        animation: topBorderShine 5s linear infinite;
    }
    @keyframes topBorderShine {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .app-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.2rem;
    }
    .app-branding {
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }
    .app-logo-box {
        width: 56px;
        height: 56px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.2), rgba(139, 92, 246, 0.2));
        border: 1.5px solid rgba(0, 229, 255, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        animation: logoFloat 4s ease-in-out infinite;
    }
    @keyframes logoFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    .app-title-text {
        font-family: var(--font-display);
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(120deg, #FFFFFF 20%, #A5F3FC 60%, #E0E7FF 95%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
    }
    .app-subtitle-text {
        color: var(--text-secondary);
        font-size: 0.88rem;
        font-weight: 400;
        margin-top: 0.35rem;
        letter-spacing: 0.015em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-status-badges {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid var(--border-subtle);
        color: var(--text-secondary);
        backdrop-filter: blur(8px);
        transition: all 0.25s ease;
    }
    .header-badge:hover {
        border-color: rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        transform: translateY(-1px);
    }
    .header-badge.highlight {
        background: rgba(0, 229, 255, 0.1);
        border-color: rgba(0, 229, 255, 0.4);
        color: var(--cyan-primary);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.15);
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
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.75); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* ══════ 顶部系统遥测指标横幅 (Telemetry Ribbon) ══════ */
    .system-metrics-ribbon {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.9rem;
        margin-top: 1.3rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .ribbon-item {
        background: rgba(11, 17, 32, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .ribbon-item:hover {
        background: rgba(16, 25, 48, 0.85);
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4), 0 0 16px rgba(0, 229, 255, 0.1);
    }
    .ribbon-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .ribbon-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }
    .ribbon-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: var(--font-mono);
        margin-top: 0.1rem;
    }

    /* ══════ 悬浮胶囊标签栏 (Floating Segmented Tab Bar) ══════ */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: flex-start !important;
        gap: 10px !important;
        background: rgba(13, 19, 33, 0.85) !important;
        padding: 6px !important;
        border-radius: 16px !important;
        border: 1px solid var(--border-subtle) !important;
        margin-bottom: 1.5rem !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.6rem !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.18) 0%, rgba(139, 92, 246, 0.18) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 22px rgba(0, 229, 255, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ══════ 全息 AI 视觉扫描观测舱 (Holographic HUD Scanning Chamber) ══════ */
    .hud-chamber {
        position: relative;
        background: radial-gradient(120% 120% at 50% 50%, rgba(15, 23, 42, 0.85) 0%, rgba(6, 9, 18, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.28);
        border-radius: 16px;
        padding: 1.1rem;
        overflow: hidden;
        box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.75), inset 0 0 24px rgba(0, 229, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .hud-chamber:hover {
        border-color: rgba(0, 229, 255, 0.55);
        box-shadow: 0 18px 48px -8px rgba(0, 0, 0, 0.85), 0 0 28px rgba(0, 229, 255, 0.2);
    }
    /* HUD 拐角瞄准框 */
    .hud-bracket {
        position: absolute;
        width: 14px;
        height: 14px;
        pointer-events: none;
        z-index: 10;
    }
    .hud-bracket-tl { top: 12px; left: 12px; border-top: 2.5px solid #00E5FF; border-left: 2.5px solid #00E5FF; }
    .hud-bracket-tr { top: 12px; right: 12px; border-top: 2.5px solid #00E5FF; border-right: 2.5px solid #00E5FF; }
    .hud-bracket-bl { bottom: 12px; left: 12px; border-bottom: 2.5px solid #00E5FF; border-left: 2.5px solid #00E5FF; }
    .hud-bracket-br { bottom: 12px; right: 12px; border-bottom: 2.5px solid #00E5FF; border-right: 2.5px solid #00E5FF; }
    
    /* 动态激光扫描线 */
    .hud-laser-scanner {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 4%, #00E5FF 30%, #FFFFFF 50%, #00E5FF 70%, transparent 96%);
        box-shadow: 0 0 14px #00E5FF, 0 0 6px #FFFFFF;
        pointer-events: none;
        z-index: 9;
        animation: laserSweep 3.2s ease-in-out infinite;
    }
    @keyframes laserSweep {
        0% { top: 6%; opacity: 0.1; }
        15% { opacity: 0.95; }
        85% { opacity: 0.95; }
        100% { top: 92%; opacity: 0.1; }
    }
    
    /* HUD 顶部遥测数据条 */
    .hud-telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: #64748B;
        padding-bottom: 0.5rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.08);
    }
    .hud-tag {
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.28);
        color: #00E5FF;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.68rem;
        letter-spacing: 0.04em;
    }

    /* ══════ Bento Grid 高级卡片系统 ══════ */
    .sci-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px -8px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .sci-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(0, 229, 255, 0.35);
        box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.65), 0 0 24px -4px var(--cyan-glow);
        transform: translateY(-2px);
    }
    .sci-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.16), transparent);
    }

    /* ══════ 视觉相框与悬浮微动效 ══════ */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: #080C14 !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.7) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        max-height: 380px !important;
        object-fit: contain !important;
    }
    div[data-testid="stImage"] img:hover {
        transform: translateY(-2px) scale(1.015) !important;
        border-color: rgba(0, 229, 255, 0.5) !important;
        box-shadow: 0 16px 32px -6px rgba(0, 0, 0, 0.85), 0 0 24px -4px rgba(0, 229, 255, 0.35) !important;
    }
    div[data-testid="stImage"] [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        margin-top: 0.5rem !important;
    }

    /* ══════ 区域标头 ══════ */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-family: var(--font-display);
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 1.4rem 0 0.9rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .section-title-icon {
        color: var(--cyan-primary);
        font-size: 1.25rem;
    }

    /* ══════ 侧边栏重塑 ══════ */
    section[data-testid="stSidebar"] {
        background-color: #080B12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.6rem 1.1rem;
    }
    .sidebar-brand-card {
        background: linear-gradient(135deg, rgba(16, 23, 42, 0.85) 0%, rgba(9, 13, 24, 0.95) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.4rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .sidebar-brand-title {
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 1.18rem;
        color: #FFFFFF;
        margin-top: 0.5rem;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.76rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* ══════ 按钮科技感 ══════ */
    div[data-testid="stButton"] button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 0.6rem 1.4rem !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(16, 23, 42, 0.75) !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--cyan-primary) !important;
        color: #FFFFFF !important;
        background: rgba(0, 229, 255, 0.12) !important;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.6) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 22px rgba(0, 180, 216, 0.35) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00C8E5 0%, #0096C7 100%) !important;
        box-shadow: 0 6px 28px rgba(0, 229, 255, 0.55) !important;
        transform: translateY(-2px);
    }

    /* ══════ 单选与模式切换器 (Pill Radio Group) ══════ */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 10px !important;
        background: rgba(13, 19, 33, 0.8) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(16px) !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        color: #94A3B8 !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(0, 229, 255, 0.08) !important;
        border-color: rgba(0, 229, 255, 0.3) !important;
        color: #FFFFFF !important;
    }

    /* ══════ 输入框与选择框 ══════ */
    div[data-baseweb="select"] > div {
        background: #0B1120 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: var(--cyan-primary) !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.15) !important;
    }

    /* ══════ 未来感全息文件上传舱 (Dropzone Chamber) ══════ */
    div[data-testid="stFileUploader"] section {
        background: radial-gradient(120% 120% at 50% 50%, rgba(0, 229, 255, 0.05) 0%, rgba(13, 19, 33, 0.85) 100%) !important;
        border: 2px dashed rgba(0, 229, 255, 0.32) !important;
        border-radius: 16px !important;
        padding: 1.8rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: inset 0 0 25px rgba(0, 229, 255, 0.04) !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--cyan-primary) !important;
        background: radial-gradient(120% 120% at 50% 50%, rgba(0, 229, 255, 0.1) 0%, rgba(16, 24, 42, 0.95) 100%) !important;
        box-shadow: 0 0 32px rgba(0, 229, 255, 0.25), inset 0 0 35px rgba(0, 229, 255, 0.08) !important;
        transform: translateY(-2px) !important;
    }

    /* ══════ 折叠框 Expander ══════ */
    div[data-testid="stExpander"] {
        background: rgba(13, 19, 33, 0.65) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px !important;
        overflow: hidden;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 229, 255, 0.35) !important;
    }

    /* ══════ 指标微型卡片与统计单元格 ══════ */
    .metric-cell {
        background: rgba(11, 17, 32, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }
    .metric-cell:hover {
        border-color: rgba(0, 229, 255, 0.4);
        background: rgba(16, 25, 48, 0.85);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4), 0 0 16px rgba(0, 229, 255, 0.1);
    }
    .metric-cell-value {
        font-family: var(--font-mono);
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--cyan-primary);
        line-height: 1.25;
    }
    .metric-cell-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ══════ 响应式调整 ══════ */
    @media (max-width: 768px) {
        .app-title-text { font-size: 1.35rem; }
        .app-header-container { padding: 1.2rem; }
        .system-metrics-ribbon { grid-template-columns: 1fr 1fr; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.9rem; font-size: 0.85rem; }
        div[data-testid="stRadio"] div[role="radiogroup"] { flex-direction: column !important; }
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
        ⚙️ <b>禽蛋滚落动力学分析系统</b><br>
        体系架构：v2.5 Pro 智能分选版<br>
        机器视觉与多模型融合驱动
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
                <span>🔬 视觉解耦：19D 形态空间</span>
            </div>
            <div class="header-badge">
                <span>🤖 决策引擎：四算法集成</span>
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
