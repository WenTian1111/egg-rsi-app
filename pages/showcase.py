"""
展示页面 - 图像处理流水线 + 数据库多维分析
西南大学大学生创新创业训练计划项目 (S202510635378)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from utils.data_loader import (
    load_fusion_data, load_model_metrics, load_feature_importance,
    RSI_LABELS, STATIC_FEATURES_19, get_egg_image_path,
    generate_pipeline_images, get_processing_images
)


def _safe_get_column(df, candidates, default=None):
    """安全获取 DataFrame 中可能存在的候选列名之一。"""
    for col in candidates:
        if col in df.columns:
            return col
    return default


def show_showcase():
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 1.5rem;">🔬</span>
            <span style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">数据库全景展示与形态分析</span>
            <span style="background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3);
                         color: #00E5FF; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem;">
                Database & Vision Telemetry
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_fusion_data()
        if df is None or df.empty:
            st.error("❌ 无法加载数据集，请检查 data/ 目录下的数据文件。")
            return
    except Exception as e:
        st.error(f"❌ 数据加载失败: {str(e)}")
        return

    # 规范化关键列名映射
    id_col = _safe_get_column(df, ['EggID', 'egg_id'], 'egg_id')
    rsi_col = _safe_get_column(df, ['RSI_GroupNum', 'RSI'], 'RSI')

    if id_col != 'egg_id':
        df['egg_id'] = df[id_col]
    if rsi_col != 'RSI':
        df['RSI'] = df[rsi_col]

    tab1, tab2 = st.tabs([
        "🧪 智能视觉处理流水线 (Vision Pipeline)",
        "📋 样本多维画像与模型基准 (Database Explorer & Benchmark)"
    ])

    with tab1:
        _show_pipeline_tab(df)

    with tab2:
        _show_browser_tab(df)


def _show_pipeline_tab(df):
    """智能视觉处理流水线子标签"""
    st.markdown("""
    <div style="background: rgba(14, 20, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 600; color: #00E5FF; font-size: 0.95rem; margin-bottom: 0.3rem;">
            ⚙️ 计算机视觉形态解耦流水线 (Computer Vision Decomposition Pipeline)
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
            系统通过标准工业镜头采集禽蛋静态顶视图像，经由<b>灰度直方图均衡</b>、<b>自适应阈值分割与形态学去噪</b>，
            提取封闭外轮廓并拟合最小外接矩形与等效椭圆，最终解析出 19 项与滚动稳定性强相关的几何与矩特征。
        </div>
    </div>
    """, unsafe_allow_html=True)

    egg_ids = sorted(df['egg_id'].unique().tolist())
    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        selected_egg = st.selectbox(
            "选择待观测标本编号",
            egg_ids,
            key="pipeline_egg_select",
            format_func=lambda x: f"第 {x} 号鸡蛋标本"
        )
    with col_stat:
        egg_row = df[df['egg_id'] == selected_egg].iloc[0]
        rsi_val = int(egg_row.get('RSI', 1))
        risk_name, risk_col, risk_icon = RSI_LABELS.get(rsi_val, ('未知', '#94A3B8', '⚪'))
        esi_val = egg_row.get('Static_ShapeIndex_机器视觉ESI', 0)
        area_val = egg_row.get('Static_Area_像素面积', 0)
        st.markdown(f"""
        <div style="display: flex; gap: 0.8rem; align-items: center; height: 100%; padding-top: 1.4rem;">
            <div style="background: rgba(14, 20, 35, 0.8); border: 1px solid {risk_col}44; border-radius: 10px;
                        padding: 0.4rem 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>{risk_icon}</span>
                <span style="font-size: 0.82rem; color: #94A3B8;">实际评级:</span>
                <span style="font-weight: 700; color: {risk_col};">{risk_name}</span>
            </div>
            <div style="background: rgba(14, 20, 35, 0.8); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
                        padding: 0.4rem 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                <span style="font-size: 0.82rem; color: #94A3B8;">蛋形指数(ESI):</span>
                <span style="font-weight: 700; color: #00E5FF; font-family: var(--font-mono);">{esi_val:.4f}</span>
            </div>
            <div style="background: rgba(14, 20, 35, 0.8); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
                        padding: 0.4rem 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                <span style="font-size: 0.82rem; color: #94A3B8;">像素面积:</span>
                <span style="font-weight: 700; color: #FFFFFF; font-family: var(--font-mono);">{area_val:.0f} px²</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4 步流水线图像展示
    with st.spinner(f"正在实时计算标本 {selected_egg} 号的处理图谱..."):
        try:
            pipeline = generate_pipeline_images(selected_egg)
        except Exception:
            pipeline = None

    steps = [
        ('original', '01. 原始光学采集', '顶视校准色温光学图像', '#38BDF8'),
        ('grayscale', '02. 灰度矩阵转换', '单通道亮度直方图均衡', '#A78BFA'),
        ('mask', '03. 掩膜二值分割', '形态学开闭运算去杂噪', '#00E5FF'),
        ('contour', '04. 轮廓质心与包围盒', '最小外接矩形与质心偏移', '#10B981'),
    ]

    cols = st.columns(4)
    for idx, (key, title, subtitle, accent) in enumerate(steps):
        with cols[idx]:
            img_bgr = pipeline.get(key) if pipeline else None
            st.markdown(f"""
            <div style="background: rgba(14, 20, 35, 0.7); border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 12px; padding: 0.6rem; margin-bottom: 0.6rem; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: {accent};">{title}</span>
                    <span style="font-size: 0.68rem; color: #64748B; background: rgba(255,255,255,0.04);
                                 padding: 0.1rem 0.4rem; border-radius: 4px;">Step {idx+1}</span>
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-bottom: 0.5rem; text-align: left;">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

            if img_bgr is not None:
                import cv2
                img_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.image(img_rgb, use_container_width=True)
            else:
                # 降级尝试专用预存图片或占位
                contour_path = get_egg_image_path(selected_egg)
                if os.path.exists(contour_path) and key in ['contour', 'original']:
                    st.image(contour_path, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="height: 200px; background: rgba(15, 23, 42, 0.5); border: 1px dashed rgba(255,255,255,0.1);
                                border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #64748B;">
                        图谱处理中或未载入
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # 19 维特征数字仪表盘
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">📊</span>
        <span>19 维形态特征数字遥测看板 (Morphological Telemetry Dashboard)</span>
    </div>
    """, unsafe_allow_html=True)

    # 12 项基础形态几何参数
    st.markdown("##### 📐 基础几何形态参数 (12项解耦变量)")
    basic_feature_meta = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数 (ESI)', '', '短轴/长轴比值，直观反映饱满度'),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数', '', '锐端与钝端曲率偏心差异'),
        ('Static_Eccentricity_离心率', '离心率', '', '椭圆拟合焦距比值，越近0越圆'),
        ('Static_Area_像素面积', '像素面积', 'px²', '蛋体在顶视平面的投影绝对像素数'),
        ('Static_Perimeter_轮廓周长', '轮廓周长', 'px', '封闭外边界像素欧氏距离累计'),
        ('Static_MajorAxisLength_长轴像素长', '长轴像素长度', 'px', '等效椭圆第一主轴像素尺度'),
        ('Static_MinorAxisLength_短轴像素长', '短轴像素长度', 'px', '等效椭圆第二主轴像素尺度'),
        ('Static_Circularity_圆形度', '圆形度', '', '4π*面积/周长²，越接近1越规整'),
        ('Static_Solidity_坚实度', '坚实度', '', '蛋体面积与凸包面积比值'),
        ('Static_Extent_延展度', '延展度', '', '蛋体面积与外接矩形面积比值'),
        ('Static_EquivalentDiameter_等效圆直径', '等效圆直径', 'px', '相同面积圆的等效直径'),
        ('Static_MajorAxisOffsetRatio_长轴偏移率', '长轴偏移率', '', '质心相对几何中心在主轴上的位移比'),
    ]

    col_grid = st.columns(4)
    for idx, (col_name, label_cn, unit, tooltip) in enumerate(basic_feature_meta):
        val = egg_row.get(col_name, 0)
        with col_grid[idx % 4]:
            if isinstance(val, float):
                disp_val = f"{val:.4f}" if abs(val) < 1000 else f"{val:.1f}"
            else:
                disp_val = str(val)
            unit_str = f" <span style='font-size:0.7rem;color:#64748B;'>{unit}</span>" if unit else ""
            st.markdown(f"""
            <div class="metric-cell" title="{tooltip}">
                <div class="metric-cell-value">{disp_val}{unit_str}</div>
                <div class="metric-cell-label">{label_cn}</div>
            </div>
            """, unsafe_allow_html=True)

    # 7 项不变 Hu 矩矩阵
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🎯 7 阶正交不变 Hu 矩矩阵 (Hu Invariant Moments Matrix)")
    hu_keys = [f'Static_Hu{i}' for i in range(1, 8)]
    hu_cols = st.columns(7)
    for i, hu_key in enumerate(hu_keys):
        with hu_cols[i]:
            hu_val = egg_row.get(hu_key, 0)
            if isinstance(hu_val, float):
                hu_disp = f"{hu_val:.3e}" if abs(hu_val) < 0.01 else f"{hu_val:.4f}"
            else:
                hu_disp = str(hu_val)
            st.markdown(f"""
            <div style="background: rgba(14, 20, 35, 0.8); border: 1px solid rgba(255,255,255,0.06);
                        border-radius: 8px; padding: 0.6rem 0.4rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #A78BFA; font-weight: 600; margin-bottom: 0.2rem;">Hu {i+1}</div>
                <div style="font-family: var(--font-mono); font-size: 0.85rem; color: #FFFFFF; font-weight: 600;">{hu_disp}</div>
                <div style="font-size: 0.62rem; color: #64748B; margin-top: 0.2rem;">阶不变性</div>
            </div>
            """, unsafe_allow_html=True)

    # 学术科研成果画廊 (Bento 风格展开)
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("🔬 点击展开论文实验台装置与验证机理全景 (Academic Apparatus & Findings)", expanded=False):
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
        g1, g2 = st.columns(2)
        with g1:
            plat_img = os.path.join(assets_dir, 'experiment_platform.png')
            if os.path.exists(plat_img):
                st.image(plat_img, caption="图 2.1 物理滚落实验平台 — 静态机器视觉采集区 + 动态倾角滚落导轨", use_container_width=True)
            risk_img = os.path.join(assets_dir, 'risk_validation.jpg')
            if os.path.exists(risk_img):
                st.image(risk_img, caption="图 2.2 风险分级与动力学失稳关联性实测验证", use_container_width=True)
        with g2:
            road_img = os.path.join(assets_dir, 'roadmap.png')
            if os.path.exists(road_img):
                st.image(road_img, caption="图 2.3 技术路线流程图 — 多特征提取与融合分类架构", use_container_width=True)
            mech_img = os.path.join(assets_dir, 'egg_physics.png')
            if os.path.exists(mech_img):
                st.image(mech_img, caption="图 2.4 蛋体几何各向异性导致偏心力矩机制解析", use_container_width=True)


def _show_browser_tab(df):
    """样本多维画像与模型基准测试子标签"""
    st.markdown("""
    <div style="background: rgba(14, 20, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 600; color: #00E5FF; font-size: 0.95rem; margin-bottom: 0.3rem;">
            📋 样本画像探查与多算法竞技场 (Sample Profiler & ML Model Benchmark)
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
            交互式探索 90 枚标本库的多维雷达画像，对比随机森林 (RF)、梯度提升 (GBDT)、支持向量机 (SVM)
            与逻辑回归 (LR) 四种主流分类器在宏准确率与 AUC 上的表现。
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 顶部快捷过滤器
    f_col1, f_col2, f_col3 = st.columns([1.2, 1, 1.8])
    with f_col1:
        risk_filter = st.multiselect(
            "筛选风险评级",
            options=[1, 2, 3],
            default=[1, 2, 3],
            format_func=lambda x: f"{RSI_LABELS.get(x, ('未知','','⚪'))[2]} {RSI_LABELS.get(x, ('未知','',''))[0]}"
        )
    if not risk_filter:
        risk_filter = [1, 2, 3]

    filtered_df = df[df['RSI'].isin(risk_filter)]
    egg_options = sorted(filtered_df['egg_id'].unique().tolist())
    if not egg_options:
        st.warning("⚠️ 当前筛选条件下无符合样本")
        return

    with f_col2:
        selected_egg = st.selectbox(
            "选择比对标本",
            egg_options,
            format_func=lambda x: f"{x}号鸡蛋标本"
        )
    with f_col3:
        n_low = len(df[df['RSI'] == 1])
        n_mid = len(df[df['RSI'] == 2])
        n_high = len(df[df['RSI'] == 3])
        st.markdown(f"""
        <div style="display: flex; gap: 0.6rem; align-items: center; padding-top: 1.6rem;">
            <span style="font-size: 0.78rem; color: #64748B;">标本库结构:</span>
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3);
                         color: #10B981; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
                🟢 低风险: {n_low}
            </span>
            <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3);
                         color: #F59E0B; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
                🟡 中风险: {n_mid}
            </span>
            <span style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3);
                         color: #EF4444; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
                🔴 高风险: {n_high}
            </span>
        </div>
        """, unsafe_allow_html=True)

    egg_row = filtered_df[filtered_df['egg_id'] == selected_egg].iloc[0]
    rsi_level = int(egg_row.get('RSI', 1))
    r_name, r_color, r_icon = RSI_LABELS.get(rsi_level, ('未知', '#94A3B8', '⚪'))

    # 左右两栏布局：左侧标本画卷，右侧雷达图与基准对比
    left_col, right_col = st.columns([1, 1.6])

    with left_col:
        st.markdown(f"""
        <div class="sci-card" style="border-top: 3px solid {r_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF;">第 {selected_egg} 号标本详情</div>
                <div style="background: {r_color}22; border: 1px solid {r_color}; color: {r_color};
                            padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">
                    {r_icon} {r_name}
                </div>
            </div>
        """, unsafe_allow_html=True)

        img_path = get_egg_image_path(selected_egg)
        if os.path.exists(img_path):
            st.image(img_path, caption=f"标本 {selected_egg} 号 — 提取几何轮廓", use_container_width=True)
        else:
            st.info("标本轮廓图像加载中")

        st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.8rem;">
                <div style="background: rgba(255,255,255,0.03); padding: 0.5rem 0.7rem; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #64748B;">蛋形指数 (ESI)</div>
                    <div style="font-weight: 700; color: #00E5FF; font-family: var(--font-mono); font-size: 1.05rem;">
                        {egg_row.get('Static_ShapeIndex_机器视觉ESI', 0):.4f}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 0.5rem 0.7rem; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #64748B;">不对称指数</div>
                    <div style="font-weight: 700; color: #FFFFFF; font-family: var(--font-mono); font-size: 1.05rem;">
                        {egg_row.get('Static_AsymmetryIndex_不对称指数', 0):.4f}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 0.5rem 0.7rem; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #64748B;">离心率</div>
                    <div style="font-weight: 700; color: #FFFFFF; font-family: var(--font-mono); font-size: 1.05rem;">
                        {egg_row.get('Static_Eccentricity_离心率', 0):.4f}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 0.5rem 0.7rem; border-radius: 8px;">
                    <div style="font-size: 0.7rem; color: #64748B;">圆形度</div>
                    <div style="font-weight: 700; color: #FFFFFF; font-family: var(--font-mono); font-size: 1.05rem;">
                        {egg_row.get('Static_Circularity_圆形度', 0):.4f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        sub_tab1, sub_tab2 = st.tabs([
            "📡 标本多维几何雷达图 (Radar Profile)",
            "🏆 机器学习模型基准比武 (ML Benchmark & Importance)"
        ])

        with sub_tab1:
            _render_cyber_radar(egg_row, df)

        with sub_tab2:
            _render_model_benchmarks()


def _render_cyber_radar(egg_row, full_df):
    """绘制高科技暗色雷达图"""
    radar_features = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数(ESI)'),
        ('Static_AsymmetryIndex_不对称指数', '不对称性'),
        ('Static_Eccentricity_离心率', '离心率'),
        ('Static_Circularity_圆形度', '圆形度'),
        ('Static_Solidity_坚实度', '坚实度'),
        ('Static_Extent_延展度', '延展度'),
    ]

    labels = []
    norm_vals = []
    for col, label in radar_features:
        labels.append(label)
        val = egg_row.get(col, 0)
        c_min = full_df[col].min() if col in full_df.columns else 0
        c_max = full_df[col].max() if col in full_df.columns else 1
        if c_max > c_min:
            n_val = (val - c_min) / (c_max - c_min)
        else:
            n_val = 0.5
        norm_vals.append(max(0.0, min(1.0, float(n_val))))

    # 闭合雷达图
    labels.append(labels[0])
    norm_vals.append(norm_vals[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_vals,
        theta=labels,
        fill='toself',
        fillcolor='rgba(0, 229, 255, 0.22)',
        line=dict(color='#00E5FF', width=2.5),
        marker=dict(color='#00E5FF', size=7, symbol='diamond'),
        name=f"标本 {egg_row.get('egg_id', '')} 号"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(14, 20, 35, 0.5)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                color='#64748B',
                gridcolor='rgba(255, 255, 255, 0.08)',
                tickfont=dict(size=9, color='#64748B')
            ),
            angularaxis=dict(
                color='#E2E8F0',
                gridcolor='rgba(255, 255, 255, 0.08)',
                tickfont=dict(size=11, color='#E2E8F0', family='Inter, sans-serif')
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=360,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_benchmarks():
    """渲染多模型准确率柱状图与特征重要性对比"""
    metrics_df = load_model_metrics()
    if metrics_df is not None and not metrics_df.empty:
        col_m1, col_m2 = st.columns([1.2, 1])
        with col_m1:
            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#94A3B8;margin-bottom:0.3rem;'>四模型核心分类效能对比</div>", unsafe_allow_html=True)
            models = metrics_df['ModelName'].tolist()
            fig = go.Figure()

            metric_configs = [
                ('Accuracy', '#00E5FF', '准确率'),
                ('Macro_F1', '#A78BFA', '宏平均 F1'),
                ('Macro_AUC', '#10B981', '宏平均 AUC'),
            ]
            for m_key, color, label in metric_configs:
                if m_key in metrics_df.columns:
                    vals = metrics_df[m_key].tolist()
                    fig.add_trace(go.Bar(
                        name=label,
                        x=models,
                        y=vals,
                        marker=dict(color=color, cornerradius=4),
                        text=[f"{v:.1%}" for v in vals],
                        textposition='outside',
                        textfont=dict(color='#E2E8F0', size=10)
                    ))

            fig.update_layout(
                barmode='group',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.2, font=dict(color='#E2E8F0', size=10)),
                xaxis=dict(tickfont=dict(color='#E2E8F0'), gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(range=[0, 1.1], tickformat='.0%', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)'),
                margin=dict(l=10, r=10, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_m2:
            st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#94A3B8;margin-bottom:0.3rem;'>Top 8 驱动特征重要性 (RF vs GBDT)</div>", unsafe_allow_html=True)
            imp_df = load_feature_importance()
            if isinstance(imp_df, pd.DataFrame) and 'RF_Importance' in imp_df.columns:
                top8 = imp_df.head(8)
                names = top8['ShortName'].tolist()
                rf_v = top8['RF_Importance'].tolist()
                gb_v = top8['GBDT_Importance'].tolist()

                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    y=names, x=rf_v, name='随机森林 (RF)',
                    orientation='h', marker_color='#00E5FF'
                ))
                fig_imp.add_trace(go.Bar(
                    y=names, x=gb_v, name='梯度提升 (GBDT)',
                    orientation='h', marker_color='#F59E0B'
                ))
                fig_imp.update_layout(
                    barmode='group',
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", y=1.2, font=dict(color='#E2E8F0', size=9)),
                    xaxis=dict(tickformat='.0%', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(tickfont=dict(color='#E2E8F0', size=10), categoryorder='total ascending'),
                    margin=dict(l=10, r=10, t=30, b=20)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("特征重要性比对数据载入中")
    else:
        st.info("模型基准指标暂不可用")
