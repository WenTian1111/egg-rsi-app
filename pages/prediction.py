"""预测页面 - RSI风险预测"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os

from utils.data_loader import load_fusion_data, RSI_LABELS, STATIC_FEATURES_19, get_egg_image_path, predict_risk, MODEL_NAMES
from utils.feature_extraction import extract_features_from_image, process_uploaded_image


def show_prediction():
    st.markdown('<div class="section-header">🤖 算法预测流水线</div>', unsafe_allow_html=True)

    mode = st.radio("选择预测模式", ["📤 上传新图片", "🎯 快速选择"], horizontal=True, label_visibility="collapsed")

    if mode == "📤 上传新图片":
        show_upload_mode()
    else:
        show_quick_mode()


def show_upload_mode():
    st.markdown("""
    <div class="card">
    上传鸡蛋照片，系统将自动完成图像预处理、特征提取、模型预测全流程。
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("上传鸡蛋照片", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            result = process_uploaded_image(uploaded_file)

            if not result.get("success"):
                st.error(f"❌ {result.get('error', '图像处理失败')}")
                return

            steps = result.get("steps", {})
            features = result.get("features", {})

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">步骤 1: 图像预处理</h4>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><div class="metric-label">原始图像</div></div>', unsafe_allow_html=True)
                if steps.get("original") is not None:
                    st.image(steps["original"], width=280)
                else:
                    st.warning("无原始图像")
            with col2:
                st.markdown('<div class="metric-card"><div class="metric-label">灰度转换</div></div>', unsafe_allow_html=True)
                if steps.get("grayscale") is not None:
                    st.image(steps["grayscale"], width=280, clamp=True)
                else:
                    st.warning("无灰度图像")
            with col3:
                st.markdown('<div class="metric-card"><div class="metric-label">HSV掩膜</div></div>', unsafe_allow_html=True)
                if steps.get("hsv_mask") is not None:
                    st.image(steps["hsv_mask"], width=280, clamp=True)
                else:
                    st.warning("无HSV掩膜")

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">步骤 2: 轮廓可视化</h4>', unsafe_allow_html=True)

            if steps.get("contour_viz") is not None:
                st.image(steps["contour_viz"], width=400)
            else:
                st.warning("无轮廓可视化图像")

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">步骤 3: 特征提取</h4>', unsafe_allow_html=True)

            display_features_grid(features)

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">步骤 4: 模型选择</h4>', unsafe_allow_html=True)

            model_options = ['svm', 'rf', 'gbdt']
            selected_model = st.selectbox(
                "选择预测模型",
                options=model_options,
                format_func=lambda x: MODEL_NAMES.get(x, x.upper())
            )

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">步骤 5: 运行预测</h4>', unsafe_allow_html=True)

            if st.button("🚀 运行预测", type="primary", use_container_width=True):
                run_upload_prediction(features, selected_model)

        except Exception as e:
            st.error(f"❌ 处理图片时出错: {str(e)}")


def show_quick_mode():
    st.markdown("""
    <div class="card">
    从90个现有鸡蛋中选择，系统将快速完成特征提取与预测。
    </div>
    """, unsafe_allow_html=True)

    egg_ids = list(range(1, 91))
    selected_egg = st.selectbox("选择鸡蛋编号", egg_ids, format_func=lambda x: f"{x}号鸡蛋")

    col_img, col_info = st.columns([1, 2])

    with col_img:
        st.markdown("**鸡蛋轮廓图像**")
        image_path = get_egg_image_path(selected_egg)
        if image_path and os.path.exists(image_path):
            st.image(image_path, width=300)
        else:
            st.warning(f"⚠️ 图像文件不存在: {image_path}")
            return

    with col_info:
        try:
            features = extract_features_from_image(image_path)
            if features is None:
                st.error("❌ 特征提取失败")
                return

            display_features_grid(features)

            st.markdown("---")
            st.markdown('<h4 style="color: #00B4D8;">模型选择</h4>', unsafe_allow_html=True)

            model_options = ['svm', 'rf', 'gbdt']
            selected_model = st.selectbox(
                "选择预测模型",
                options=model_options,
                format_func=lambda x: MODEL_NAMES.get(x, x.upper()),
                key="quick_model_select"
            )

            st.markdown("---")
            if st.button("🚀 运行预测", type="primary", use_container_width=True, key="quick_predict_btn"):
                run_quick_prediction(selected_egg, features, selected_model)

        except Exception as e:
            st.error(f"❌ 处理时出错: {str(e)}")


def display_features_grid(features):
    basic_features = [
        ('Static_Area_像素面积', '面积', 'pixels'),
        ('Static_Perimeter_轮廓周长', '周长', 'pixels'),
        ('Static_MajorAxisLength_长轴像素长度', '长轴', 'pixels'),
        ('Static_MinorAxisLength_短轴像素长度', '短轴', 'pixels'),
        ('Static_Eccentricity_离心率', '离心率', ''),
        ('Static_ShapeIndex_机器视觉ESI', 'ESI', ''),
        ('Static_Circularity_圆形度', '圆形度', ''),
        ('Static_Solidity_坚实度', '坚实度', ''),
        ('Static_Extent_延展度', '延展度', ''),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数', ''),
    ]

    cols = st.columns(3)
    for idx, (feat_key, label, unit) in enumerate(basic_features):
        with cols[idx % 3]:
            value = features.get(feat_key, 0)
            if isinstance(value, float):
                if abs(value) < 0.01:
                    value_str = f"{value:.6f}"
                elif abs(value) < 1:
                    value_str = f"{value:.4f}"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            display_value = f"{value_str} {unit}" if unit else value_str
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{display_value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("**Hu矩特征**")
    hu_features = ['Static_Hu1', 'Static_Hu2', 'Static_Hu3', 'Static_Hu4', 'Static_Hu5', 'Static_Hu6', 'Static_Hu7']
    hu_labels = ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']
    hu_cols = st.columns(7)
    for idx, (feat_key, label) in enumerate(zip(hu_features, hu_labels)):
        with hu_cols[idx]:
            value = features.get(feat_key, 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.1rem;">{value:.4f}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def run_upload_prediction(features, model_name):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("正在预测...")
        progress_bar.progress(0.3)

        prediction, probabilities = predict_risk(features, model_name)
        if prediction is None:
            st.error("❌ 预测失败")
            return

        progress_bar.progress(0.7)
        status_text.text("正在生成结果...")
        display_prediction_result(prediction, probabilities)

        progress_bar.progress(1.0)
        status_text.text("完成!")

    except Exception as e:
        st.error(f"❌ 预测出错: {str(e)}")


def run_quick_prediction(egg_id, features, model_name):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("正在预测...")
        progress_bar.progress(0.3)

        prediction, probabilities = predict_risk(features, model_name)
        if prediction is None:
            st.error("❌ 预测失败")
            return

        progress_bar.progress(0.7)
        status_text.text("正在生成结果...")
        display_prediction_result(prediction, probabilities)

        progress_bar.progress(1.0)
        status_text.text("完成!")

    except Exception as e:
        st.error(f"❌ 预测出错: {str(e)}")


def display_prediction_result(prediction, probabilities):
    st.markdown("---")
    st.markdown('<h4 style="color: #00B4D8;">预测结果</h4>', unsafe_allow_html=True)

    label, color, icon = RSI_LABELS.get(prediction, ("未知", "#888888", "❓"))

    st.markdown(f"""
    <div class="card result-glow" style="border-left: 4px solid {color};">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 3rem;">{icon}</span>
            <div>
                <div style="font-size: 0.9rem; color: #ADB5BD;">预测风险等级</div>
                <div style="font-size: 2rem; font-weight: 700; color: {color};">{label}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h4 style="color: #00B4D8;">概率分布</h4>', unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        '风险等级': ['低风险', '中风险', '高风险'],
        '概率': probabilities
    })
    prob_df['风险等级'] = pd.Categorical(prob_df['风险等级'], categories=['低风险', '中风险', '高风险'], ordered=True)
    prob_df = prob_df.sort_values('风险等级')

    colors = ['#4ECDC4', '#FFE66D', '#FF6B6B']
    fig = go.Figure(data=[
        go.Bar(
            y=prob_df['风险等级'],
            x=prob_df['概率'],
            orientation='h',
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in prob_df['概率']],
            textposition='outside'
        )
    ])
    fig.update_layout(
        height=200,
        margin=dict(l=100, r=50, t=20, b=20),
        xaxis=dict(range=[0, 1.15], tickformat=".0%", title=None),
        yaxis=dict(title=None),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<h4 style="color: #00B4D8;">分选建议</h4>', unsafe_allow_html=True)

    advice_map = {
        1: ("🟢 低风险鸡蛋", "该鸡蛋滚落稳定性良好，建议放入<font color='#4ECDC4'>**优等品通道**</font>。", "#4ECDC4"),
        2: ("🟡 中风险鸡蛋", "该鸡蛋存在一定滚落风险，建议放入<font color='#FFE66D'>**次等品通道**</font>或进行进一步检测。", "#FFE66D"),
        3: ("🔴 高风险鸡蛋", "该鸡蛋滚落稳定性较差，建议放入<font color='#FF6B6B'>**风险品通道**</font>，不适用于滚落实验。", "#FF6B6B"),
    }

    if prediction in advice_map:
        title, content, color = advice_map[prediction]
        st.markdown(f"""
        <div class="card" style="border-left-color: {color};">
            <div style="font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</div>
            <div style="color: #ADB5BD;">{content}</div>
        </div>
        """, unsafe_allow_html=True)