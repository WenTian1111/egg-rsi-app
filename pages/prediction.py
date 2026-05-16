"""预测页面 - V4 完整重写（修复文字大小 + 增强UI）"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cv2
import os
from PIL import Image
import numpy as np

from utils.data_loader import load_fusion_data, RSI_LABELS, STATIC_FEATURES_19, \
    get_egg_image_path, predict_risk, MODEL_NAMES
from utils.feature_extraction import extract_features_from_image, process_uploaded_image


def display_prediction_result(prediction, probabilities):
    """显示预测结果：风险徽章 + 概率条形图 + 排序建议"""
    try:
        risk_label, risk_color, risk_emoji = RSI_LABELS.get(prediction, ('未知', '#888888', '⚪'))
    except Exception:
        risk_label, risk_color, risk_emoji = '未知', '#888888', '⚪'

    col1, col2 = st.columns([1, 2], gap='medium')

    with col1:
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg, {risk_color}22, {risk_color}44);
            border: 2px solid {risk_color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 0 20px {risk_color}55;
        ">
            <div style="font-size: 48px; margin-bottom: 8px;">{risk_emoji}</div>
            <div style="font-size: 28px; font-weight: bold; color: {risk_color}; margin-bottom: 4px;">
                {risk_label}
            </div>
            <div style="font-size: 14px; color: #888;">风险等级 {prediction}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        prob_df = pd.DataFrame({
            '风险等级': ['低风险 🟢', '中风险 🟡', '高风险 🔴'],
            '概率': [probabilities[0], probabilities[1], probabilities[2]]
        })

        fig = go.Figure()
        colors = ['#4ECDC4', '#FFE66D', '#FF6B6B']
        for i, row in prob_df.iterrows():
            fig.add_trace(go.Bar(
                y=[row['风险等级']],
                x=[row['概率']],
                orientation='h',
                marker_color=colors[i],
                text=f"{row['概率']:.1%}",
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white' if i == 2 else 'black', size=14, family='Arial Black'),
                showlegend=False,
                hovertemplate=f"{row['风险等级']}: {row['概率']:.1%}<extra></extra>"
            ))

        fig.update_layout(
            title=dict(text='📊 预测概率分布', font=dict(size=16), x=0.5),
            xaxis=dict(title='概率', range=[0, 1], tickformat='.0%'),
            yaxis=dict(categoryorder='array', categoryarray=['高风险 🔴', '中风险 🟡', '低风险 🟢']),
            height=180,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, width='stretch')

    # Advice card
    advice_map = {
        1: ('✅ 低风险建议', '鸡蛋品质良好，建议优先上架销售', '#4ECDC4'),
        2: ('⚠️ 中风险建议', '建议进行进一步检测或降价处理', '#FFE66D'),
        3: ('🚨 高风险建议', '建议降级处理或拒绝收购', '#FF6B6B')
    }
    advice_title, advice_text, advice_color = advice_map.get(prediction, ('❓ 建议', '无法确定建议', '#888888'))

    st.markdown(f'''
    <div style="
        background: linear-gradient(135deg, {advice_color}11, {advice_color}22);
        border-left: 4px solid {advice_color};
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
    ">
        <div style="font-size: 16px; font-weight: bold; color: {advice_color}; margin-bottom: 8px;">
            {advice_title}
        </div>
        <div style="font-size: 14px; color: #888;">
            {advice_text}
        </div>
    </div>
    ''', unsafe_allow_html=True)


def _mini_metric(label, value, unit=''):
    """Compact metric display with small font."""
    if isinstance(value, float):
        if abs(value) < 0.001:
            display = f"{value:.4e}"
        elif abs(value) < 0.01:
            display = f"{value:.6f}"
        elif value >= 10000:
            display = f"{value:.1f}"
        else:
            display = f"{value:.4f}"
    else:
        display = str(value)
    if unit:
        display += f' <span style="color:#666;font-size:0.7rem;">{unit}</span>'
    st.markdown(f"""
    <div style="
        background: #1A1C23;
        border: 1px solid #2D2D3D;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
        margin: 4px 0;
    ">
        <div style="color: #00B4D8; font-size: 1.0rem; font-weight: 700; line-height: 1.3;">
            {display}
        </div>
        <div style="color: #ADB5BD; font-size: 0.7rem; font-weight: 500; margin-top: 1px;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_features_grid(features):
    """显示19个特征的紧凑网格布局（V4: 小字号自定义卡片，替代st.metric）"""
    try:
        if features is None:
            st.warning('无可用特征数据')
            return

        # Basic features (12)
        basic_features = [
            ('Static_Area_像素面积', '像素面积', 'px²'),
            ('Static_Perimeter_轮廓周长', '轮廓周长', 'px'),
            ('Static_MajorAxisLength_长轴像素长度', '长轴', 'px'),
            ('Static_MinorAxisLength_短轴像素长度', '短轴', 'px'),
            ('Static_EquivalentDiameter_等效圆直径', '等效直径', 'px'),
            ('Static_Eccentricity_离心率', '离心率', ''),
            ('Static_ShapeIndex_机器视觉ESI', 'ESI', ''),
            ('Static_Circularity_圆形度', '圆形度', ''),
            ('Static_Solidity_坚实度', '坚实度', ''),
            ('Static_Extent_延展度', '延展度', ''),
            ('Static_AsymmetryIndex_不对称指数', '不对称指数', ''),
            ('Static_MajorAxisOffsetRatio_长轴偏移率', '长轴偏移率', ''),
        ]

        st.markdown('### 📐 基础形态特征')
        cols = st.columns(4)
        for idx, (key, label, unit) in enumerate(basic_features):
            with cols[idx % 4]:
                value = features.get(key, 0)
                _mini_metric(label, value, unit)

        # Hu moments (7)
        hu_features = [
            ('Static_Hu1', 'Hu1'), ('Static_Hu2', 'Hu2'), ('Static_Hu3', 'Hu3'),
            ('Static_Hu4', 'Hu4'), ('Static_Hu5', 'Hu5'), ('Static_Hu6', 'Hu6'),
            ('Static_Hu7', 'Hu7'),
        ]
        st.markdown('### 🎯 Hu矩特征')
        cols = st.columns(7)
        for idx, (key, label) in enumerate(hu_features):
            with cols[idx]:
                value = features.get(key, 0)
                if isinstance(value, float):
                    if abs(value) < 0.001:
                        display = f"{value:.4e}"
                    else:
                        display = f"{value:.4f}"
                else:
                    display = str(value)
                st.markdown(f"""
                <div style="text-align:center;padding:6px 4px;background:#1A1C23;border-radius:8px;border:1px solid #2D2D3D;margin:2px;">
                    <div style="color:#00B4D8;font-size:0.85rem;font-weight:700;">{display}</div>
                    <div style="color:#ADB5BD;font-size:0.65rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'显示特征网格时出错: {str(e)}')


def show_prediction():
    st.markdown('<div class="section-header">🤖 鸡蛋风险预测</div>', unsafe_allow_html=True)

    try:
        mode = st.radio(
            '选择预测模式',
            ['📤 上传新图片', '🎯 快速选择'],
            horizontal=True,
            label_visibility='collapsed'
        )

        if mode == '📤 上传新图片':
            _upload_mode()
        else:
            _quick_select_mode()

    except Exception as e:
        st.error(f'预测页面加载出错: {str(e)}')


def _upload_mode():
    uploaded_file = st.file_uploader(
        '上传鸡蛋图片',
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help='支持 PNG、JPG、BMP 格式'
    )

    if uploaded_file is not None:
        with st.spinner('🔄 正在处理图像...'):
            result = process_uploaded_image(uploaded_file)

        if result.get('success'):
            st.session_state['upload_result'] = result

            # Show pipeline steps
            st.markdown("#### 📷 图像处理流水线")
            cols = st.columns(4)
            step_info = [
                ('original', '原图'),
                ('grayscale', '灰度图'),
                ('hsv_mask', '分割掩膜'),
                ('contour_viz', '轮廓可视化'),
            ]
            for idx, (key, label) in enumerate(step_info):
                with cols[idx]:
                    img = result['steps'].get(key)
                    if img is not None:
                        st.image(img, width='stretch', caption=label)
                    else:
                        st.info(f"{label}: N/A")

            st.markdown('---')
            # Show which segmentation strategy was used
            strategy = result.get('strategy', '未知')
            strategy_icons = {
                'L*a*b*': '🎨 L*a*b* 色彩空间分割',
                '灰度Otsu': '⚪ Otsu 阈值分割',
                'GrabCut中心': '🖌️ GrabCut 中心矩形分割',
                'Canny': '✏️ Canny 边缘检测分割',
                '腐蚀法': '🔲 渐进腐蚀分割',
            }
            strategy_icon = strategy_icons.get(strategy, f'🔧 {strategy} 分割')
            st.caption(f"分割策略：{strategy_icon}")
            if result.get('warning'):
                st.caption(f"⚠️ {result['warning']}")
            
            display_features_grid(result.get('features'))

            st.markdown('---')
            st.markdown('### 🤖 模型选择与预测')

            model_options = list(MODEL_NAMES.keys())
            default_idx = model_options.index('svm') if 'svm' in model_options else 0
            selected_model = st.selectbox(
                '选择预测模型', model_options,
                index=default_idx,
                format_func=lambda x: MODEL_NAMES.get(x, x),
                key='upload_model_select'
            )

            if st.button('🚀 运行预测', type='primary', width='stretch', key='upload_predict_btn'):
                features = result.get('features')
                if features:
                    with st.spinner('🧠 正在预测...'):
                        prediction, probabilities = predict_risk(features, selected_model)
                    if prediction is not None:
                        st.session_state['upload_prediction'] = (prediction, probabilities)
                        st.session_state['upload_model'] = selected_model
                    else:
                        st.error('预测失败，请检查模型是否已训练')
                else:
                    st.error('无法获取特征，请重新上传图片')

            if 'upload_prediction' in st.session_state:
                pred, probs = st.session_state['upload_prediction']
                st.markdown('---')
                st.markdown('### 📋 预测结果')
                display_prediction_result(pred, probs)
        else:
            st.error(f"❌ 图像处理失败: {result.get('error', '未知错误')}")


def _quick_select_mode():
    st.markdown('### 🎯 快速选择鸡蛋')

    col1, col2 = st.columns([1, 2])

    with col1:
        # Only show eggs that have images
        all_eggs = sorted([int(f.split('号')[0]) for f in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'egg_images')) if f.endswith('.jpg')])
        egg_id = st.selectbox(
            '选择鸡蛋编号', all_eggs,
            format_func=lambda x: f'{x}号鸡蛋',
            key='quick_egg_select'
        )

        image_path = get_egg_image_path(egg_id)
        if os.path.exists(image_path):
            st.image(image_path, width='stretch', caption=f'{egg_id}号鸡蛋轮廓')
        else:
            st.warning(f'未找到鸡蛋图像')

        # Quick info
        fusion_df = load_fusion_data()
        if fusion_df is not None:
            egg_row = fusion_df[fusion_df['EggID'] == egg_id]
            if not egg_row.empty:
                risk_level = egg_row.iloc[0].get('RSI_GroupNum', 1)
                risk_label, risk_color, risk_emoji = RSI_LABELS.get(risk_level, ('未知', '#888', '⚪'))
                st.markdown(f"""
                <div style="text-align:center;padding:10px;background:{risk_color}22;border-radius:8px;border:1px solid {risk_color};margin-top:8px;">
                    <span style="font-size:24px;">{risk_emoji}</span>
                    <span style="color:{risk_color};font-weight:bold;font-size:16px;margin-left:8px;">{risk_label}</span>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        # Extract features from contour image (same pipeline as uploaded photos)
        features = extract_features_from_image(image_path) if os.path.exists(image_path) else None
        display_features_grid(features)
        st.session_state['quick_features'] = features
        st.session_state['quick_egg_id'] = egg_id

    st.markdown('---')

    model_options = list(MODEL_NAMES.keys())
    default_idx = model_options.index('svm') if 'svm' in model_options else 0
    selected_model = st.selectbox(
        '选择预测模型', model_options,
        index=default_idx,
        format_func=lambda x: MODEL_NAMES.get(x, x),
        key='quick_model_select'
    )

    if st.button('🚀 运行预测', type='primary', width='stretch', key='quick_predict_btn'):
        features = st.session_state.get('quick_features')
        if features:
            with st.spinner('🧠 正在预测...'):
                prediction, probabilities = predict_risk(features, selected_model)
            if prediction is not None:
                st.session_state['quick_prediction'] = (prediction, probabilities)
                st.session_state['quick_model'] = selected_model
            else:
                st.error('预测失败，请检查模型是否已训练')
        else:
            st.error('无法获取特征，请重新选择鸡蛋')

    if 'quick_prediction' in st.session_state:
        pred, probs = st.session_state['quick_prediction']
        st.markdown('---')
        st.markdown('### 📋 预测结果')
        display_prediction_result(pred, probs)
