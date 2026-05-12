"""展示页面 - V4 完整重写"""
import streamlit as st
from utils.data_loader import (
    load_fusion_data, load_model_metrics, load_feature_importance,
    RSI_LABELS, STATIC_FEATURES_19, get_egg_image_path,
    generate_pipeline_images, FEATURE_PIPELINE_STEPS
)
import pandas as pd
import plotly.graph_objects as go
import os
import cv2
from PIL import Image
import numpy as np


def show_showcase():
    st.markdown('<div class="section-header">🔬 数据库展示与分析</div>', unsafe_allow_html=True)

    fusion_df = load_fusion_data()
    if fusion_df is None or fusion_df.empty:
        st.error("❌ 无法加载融合数据集，请检查 data/ 目录下的数据文件。")
        return

    tab1, tab2 = st.tabs(["🧪 图像处理流水线", "📋 数据库浏览"])

    with tab1:
        _show_processing_pipeline(fusion_df)

    with tab2:
        _show_database_browser(fusion_df)


def _show_processing_pipeline(fusion_df):
    st.markdown("### 🧪 图像处理流水线")
    st.markdown("""
    <div class="info-box">
        选择任意鸡蛋样本，系统将实时生成完整的图像处理流水线可视化：<br>
        <b>原图 → 灰度 → 二值掩膜 → 轮廓+质心+包围盒</b>
    </div>
    """, unsafe_allow_html=True)

    egg_ids = sorted(fusion_df['egg_id'].unique().tolist())
    selected_egg = st.selectbox("选择鸡蛋样本", egg_ids, key="pipeline_egg_selector",
                                format_func=lambda x: f"第 {x} 号鸡蛋")

    # Generate pipeline images dynamically
    with st.spinner(f"正在为第 {selected_egg} 号鸡蛋生成流水线..."):
        pipeline = generate_pipeline_images(selected_egg)

    if pipeline is None:
        st.warning(f"第 {selected_egg} 号鸡蛋的图像文件缺失，无法生成流水线。")
        return

    # Display 4 steps in 2x2 grid
    step_keys = ['original', 'grayscale', 'mask', 'contour']
    step_labels = ['📷 原图', '⚪ 灰度图', '⬛ 二值掩膜', '🔵 轮廓+质心+包围盒']
    step_colors = ['#4ECDC4', '#95E1D3', '#7C3AED', '#FF6B6B']

    cols = st.columns(2)
    for idx, (key, label, color) in enumerate(zip(step_keys, step_labels, step_colors)):
        with cols[idx % 2]:
            img_bgr = pipeline.get(key)
            if img_bgr is not None:
                img_rgb = cv2_cvt(img_bgr)
                st.image(img_rgb, width='stretch')
            else:
                st.image("https://placehold.co/400x300?text=Step+Not+Available", width='stretch')

    # Features for this egg
    st.markdown("#### 📊 提取特征")
    egg_row = fusion_df[fusion_df['egg_id'] == selected_egg].iloc[0]

    # Feature metrics in 3-column grid - compact style
    features_display = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数 ESI'),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数'),
        ('Static_Eccentricity_离心率', '离心率'),
        ('Static_Area_像素面积', '像素面积 (px²)'),
        ('Static_Perimeter_轮廓周长', '轮廓周长 (px)'),
        ('Static_Circularity_圆形度', '圆形度'),
        ('Static_Solidity_坚实度', '坚实度'),
        ('Static_Extent_延展度', '延展度'),
        ('Static_MajorAxisLength_长轴像素长度', '长轴 (px)'),
        ('Static_MinorAxisLength_短轴像素长度', '短轴 (px)'),
        ('Static_EquivalentDiameter_等效圆直径', '等效直径 (px)'),
        ('Static_MajorAxisOffsetRatio_长轴偏移率', '长轴偏移率'),
    ]

    cols = st.columns(4)
    for idx, (col_name, label) in enumerate(features_display):
        with cols[idx % 4]:
            try:
                value = egg_row.get(col_name, 0)
                _mini_metric(label, value)
            except Exception:
                _mini_metric(label, 0)

    # Hu moments table
    st.markdown("#### 📐 Hu矩特征")
    hu_cols = ['Static_Hu1', 'Static_Hu2', 'Static_Hu3', 'Static_Hu4',
               'Static_Hu5', 'Static_Hu6', 'Static_Hu7']
    hu_data = [egg_row.get(c, 0) for c in hu_cols]
    hu_df = pd.DataFrame({
        'Hu矩': ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7'],
        '数值': [f"{v:.4e}" if abs(v) < 0.001 else f"{v:.4f}" for v in hu_data]
    })
    st.table(hu_df)

    # Risk prediction badge
    risk_level = egg_row.get('RSI', 1)
    risk_label, risk_color, risk_emoji = RSI_LABELS.get(risk_level, ('未知', '#888', '⚪'))
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {risk_color}22, {risk_color}44);
        border: 2px solid {risk_color};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    ">
        <div style="font-size: 36px;">{risk_emoji}</div>
        <div style="font-size: 20px; font-weight: bold; color: {risk_color};">
            风险等级: {risk_label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _show_database_browser(fusion_df):
    st.markdown("### 📋 数据库浏览")

    # Sidebar filters
    with st.sidebar:
        st.markdown("#### 🔍 筛选条件")
        risk_options = sorted(fusion_df['RSI'].unique().tolist())
        selected_risks = st.multiselect(
            "风险等级",
            options=risk_options,
            default=risk_options,
            format_func=lambda x: f"{RSI_LABELS.get(x, ('未知', '#888', '⚪'))[2]} {RSI_LABELS.get(x, ('未知', '#888', '⚪'))[0]}"
        )

        filtered_df = fusion_df[fusion_df['RSI'].isin(selected_risks)]
        egg_options = sorted(filtered_df['egg_id'].unique().tolist())
        if not egg_options:
            st.warning("当前筛选条件下无鸡蛋数据")
            return
        selected_egg = st.selectbox("选择鸡蛋", egg_options,
                                    format_func=lambda x: f"第 {x} 号鸡蛋")

    if filtered_df.empty:
        st.info("当前筛选条件下无数据")
        return

    egg_row = filtered_df[filtered_df['egg_id'] == selected_egg].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        # Egg image
        img_path = get_egg_image_path(selected_egg)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, width='stretch', caption=f"第 {selected_egg} 号鸡蛋轮廓图")
        else:
            st.image("https://placehold.co/300x300?text=Contour+Not+Found", width='stretch')

        # Basic info card
        risk_level = egg_row.get('RSI', 1)
        risk_label, risk_color, risk_emoji = RSI_LABELS.get(risk_level, ('未知', '#888', '⚪'))
        st.markdown(f"""
        <div class="card">
            <p style="margin: 4px 0;"><b>📌 编号:</b> 第 {egg_row.get('egg_id', '?')} 号</p>
            <p style="margin: 4px 0;"><b>{risk_emoji} 风险:</b>
                <span style="color:{risk_color};font-weight:bold;">{risk_label}</span>
            </p>
            <p style="margin: 4px 0;"><b>📐 ESI:</b> {egg_row.get('Static_ShapeIndex_机器视觉ESI', 0):.4f}</p>
            <p style="margin: 4px 0;"><b>📏 面积:</b> {egg_row.get('Static_Area_像素面积', 0):.0f} px²</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        tab_radar, tab_model = st.tabs(["📡 特征雷达图", "📊 模型对比"])

        with tab_radar:
            _show_radar(egg_row)

        with tab_model:
            _show_model_comparison()


def _show_radar(egg_row):
    st.markdown("##### 特征雷达图")
    radar_features = [
        ('Static_ShapeIndex_机器视觉ESI', 'ESI'),
        ('Static_AsymmetryIndex_不对称指数', '不对称性'),
        ('Static_Eccentricity_离心率', '离心率'),
        ('Static_Circularity_圆形度', '圆形度'),
        ('Static_Solidity_坚实度', '坚实度'),
        ('Static_Extent_延展度', '延展度'),
    ]

    radar_labels = []
    radar_values = []
    for col_name, label in radar_features:
        raw_val = egg_row.get(col_name, 0)
        # Normalize to [0, 1] range (empirical bounds)
        bounds = {'Static_ShapeIndex_机器视觉ESI': (30, 100),
                  'Static_AsymmetryIndex_不对称指数': (0, 0.5),
                  'Static_Eccentricity_离心率': (0, 1),
                  'Static_Circularity_圆形度': (0, 1),
                  'Static_Solidity_坚实度': (0, 1),
                  'Static_Extent_延展度': (0, 1)}
        lo, hi = bounds.get(col_name, (0, 1))
        normalized = max(0, min(1, (raw_val - lo) / (hi - lo))) if hi > lo else 0
        radar_labels.append(label)
        radar_values.append(normalized)

    radar_values += radar_values[:1]
    radar_labels += radar_labels[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=radar_values,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(color='#4ECDC4', size=8)
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(color='white', gridcolor='rgba(255,255,255,0.2)'),
            angularaxis=dict(color='white')
        ),
        margin=dict(l=40, r=40, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')


def _show_model_comparison():
    st.markdown("##### 模型性能对比")
    try:
        metrics_df = load_model_metrics()
        if metrics_df is not None and not metrics_df.empty:
            model_names = metrics_df['ModelName'].tolist()

            tab_bar, tab_importance = st.tabs(["📊 性能对比", "⭐ 特征重要性"])

            with tab_bar:
                fig = go.Figure()
                colors = {'Accuracy': '#4ECDC4', 'Macro_F1': '#FFE66D', 'Macro_AUC': '#FF6B6B'}
                for metric in ['Accuracy', 'Macro_F1', 'Macro_AUC']:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=model_names,
                        y=metrics_df[metric].tolist(),
                        text=[f"{v:.3f}" for v in metrics_df[metric].tolist()],
                        textposition='outside',
                        marker_color=colors.get(metric, '#4ECDC4'),
                    ))

                fig.update_layout(
                    height=350,
                    barmode='group',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(title="指标", orientation="h", y=1.1),
                    xaxis_title="模型",
                    yaxis_title="分数",
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                fig.update_xaxes(color='white')
                fig.update_yaxes(color='white', range=[0, 1.1])
                st.plotly_chart(fig, width='stretch')

            with tab_importance:
                _show_feature_importance()
        else:
            st.info("模型指标数据不可用")
    except Exception as e:
        st.warning(f"模型对比加载失败: {e}")


def _show_feature_importance():
    """Show RF + GBDT feature importance side-by-side."""
    try:
        imp = load_feature_importance()
        if imp is None:
            st.info("特征重要性数据不可用")
            return

        # New consolidated format: ShortName,FullName,RF_Importance,GBDT_Importance
        if 'RF_Importance' in imp.columns and 'GBDT_Importance' in imp.columns:
            st.markdown("**随机森林 vs GBDT 特征重要性对比**")

            # Sort by RF importance descending, take top 15
            top = imp.sort_values('RF_Importance', ascending=False).head(15)
            names = top['ShortName'].tolist()
            rf_vals = top['RF_Importance'].tolist()
            gbdt_vals = top['GBDT_Importance'].tolist()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='RF (随机森林)',
                y=names,
                x=rf_vals,
                orientation='h',
                marker_color='#4ECDC4',
                text=[f"{v:.1%}" for v in rf_vals],
                textposition='outside',
            ))
            fig.add_trace(go.Bar(
                name='GBDT (梯度提升)',
                y=names,
                x=gbdt_vals,
                orientation='h',
                marker_color='#FFE66D',
                text=[f"{v:.1%}" for v in gbdt_vals],
                textposition='outside',
            ))

            fig.update_layout(
                height=500,
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(title="模型", orientation="h", y=1.08),
                xaxis_title="重要性",
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(categoryorder='array', categoryarray=names[::-1]),
                margin=dict(l=10, r=80, t=40, b=20),
            )
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')
            st.plotly_chart(fig, width='stretch')

            # Show raw data table
            with st.expander("📊 查看完整数据表"):
                display_df = top[['ShortName', 'RF_Importance', 'GBDT_Importance']].copy()
                display_df['RF_Importance'] = display_df['RF_Importance'].apply(lambda x: f"{x:.4f}")
                display_df['GBDT_Importance'] = display_df['GBDT_Importance'].apply(lambda x: f"{x:.4f}")
                display_df.columns = ['特征', 'RF 重要性', 'GBDT 重要性']
                st.dataframe(display_df, width='stretch')
        else:
            # Fallback: old format (single model)
            rf_imp, gbdt_imp = imp if isinstance(imp, tuple) else (imp, None)
            if rf_imp is not None:
                st.markdown("**随机森林特征重要性**")
                top_features = rf_imp.head(10)
                x_vals = top_features.iloc[:, 0].tolist() if len(top_features.columns) > 1 else top_features.index.tolist()
                y_vals = top_features.iloc[:, 1].tolist() if len(top_features.columns) > 1 else top_features.values.tolist()

                fig = go.Figure(data=[go.Bar(
                    x=x_vals, y=y_vals,
                    marker_color='#4ECDC4',
                )])
                fig.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="特征",
                    yaxis_title="重要性",
                    margin=dict(l=40, r=40, t=10, b=80),
                )
                fig.update_xaxes(color='white', tickangle=45)
                fig.update_yaxes(color='white')
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("特征重要性数据不可用")
    except Exception as e:
        st.warning(f"特征重要性加载失败: {e}")


def _mini_metric(label, value):
    """Compact metric card with small font."""
    if isinstance(value, float):
        if abs(value) < 0.01:
            display = f"{value:.6f}"
        elif value >= 1000:
            display = f"{value:.1f}"
        else:
            display = f"{value:.4f}"
    else:
        display = str(value)
    st.markdown(f"""
    <div class="metric-card" style="padding: 8px 12px;">
        <div style="color: #00B4D8; font-size: 1.1rem; font-weight: 700; line-height: 1.3;">
            {display}
        </div>
        <div style="color: #ADB5BD; font-size: 0.75rem; font-weight: 500; margin-top: 2px;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def cv2_cvt(bgr_img):
    """Convert cv2 BGR image to RGB for PIL/Streamlit."""
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
