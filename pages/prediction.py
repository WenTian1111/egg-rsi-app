"""
预测页面 - 鸡蛋滚落风险智能预测与分选决策
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from utils.data_loader import (
    load_fusion_data, RSI_LABELS, STATIC_FEATURES_19,
    get_egg_image_path, predict_risk, MODEL_NAMES
)
from utils.feature_extraction import extract_features_from_image, process_uploaded_image


def show_prediction():
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 1.5rem;">🤖</span>
            <span style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">鸡蛋滚落稳定性智能预测与分选决策</span>
            <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.35);
                         color: #C084FC; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem;">
                AI Sorting & Diagnostic Studio
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "选择预测输入来源",
        ["📤 上传真实拍摄照片实时推理 (Live Upload)", "🎯 从 90 枚标准化标本库快速调用 (Database Benchmark)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    if "Live Upload" in mode:
        _render_upload_workflow()
    else:
        _render_quick_select_workflow()

    # 底部学术验证与聚类分析视窗
    _render_research_insights()


def _render_upload_workflow():
    """上传拍摄照片的工作流"""
    st.markdown("""
    <div style="background: rgba(14, 20, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 600; color: #00E5FF; font-size: 0.95rem; margin-bottom: 0.3rem;">
            📤 自定义图像采集与全自动化特征提取
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem;">
            请上传禽蛋顶视拍摄图像（建议在纯色或高对比背景下拍摄）。系统将自动触发视觉分割流水线，
            解耦出 19 维静态特征并调用已训练的多分类机器学习核心进行稳定性风险推断。
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "上传禽蛋图像 (支持 PNG, JPG, JPEG, BMP)",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="推荐使用自顶向下正交拍摄的清晰鸡蛋照片"
    )

    if uploaded_file is not None:
        with st.spinner("🔄 计算机视觉流水线处理中 (色彩空间转换 ➔ 掩膜分割 ➔ 轮廓解耦)..."):
            result = process_uploaded_image(uploaded_file)

        if result.get('success'):
            st.session_state['upload_result'] = result
            strategy = result.get('strategy', '自适应阈值')

            # 流水线4步图示
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 1.2rem 0 0.8rem;
                        background: rgba(11, 17, 32, 0.6); padding: 0.6rem 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="display: flex; align-items: center; gap: 0.5rem; font-family: var(--font-mono); font-size: 0.8rem;">
                    <span style="color: #00E5FF; font-weight: 700;">VISION PIPELINE:</span>
                    <span style="color: #E2E8F0;">LIVE SEGMENTATION COMPLETE</span>
                </div>
                <div style="background: rgba(0, 229, 255, 0.12); border: 1px solid rgba(0, 229, 255, 0.35);
                             color: #00E5FF; padding: 0.18rem 0.65rem; border-radius: 9999px; font-size: 0.74rem; font-weight: 600; font-family: var(--font-mono);">
                    ENGINE: {strategy}
                </div>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(4)
            step_defs = [
                ('original', '01. 原始图像', '#38BDF8'),
                ('grayscale', '02. 灰度矩阵', '#A78BFA'),
                ('hsv_mask', '03. 分割掩膜', '#00E5FF'),
                ('contour_viz', '04. 拟合轮廓', '#10B981'),
            ]
            for idx, (key, label, accent_c) in enumerate(step_defs):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background: rgba(13, 19, 33, 0.8); border: 1px solid rgba(255, 255, 255, 0.08);
                                border-top: 2.5px solid {accent_c}; border-radius: 12px; padding: 0.55rem 0.75rem; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #FFFFFF;">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    img = result['steps'].get(key)
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        st.info(f"{label}: 处理中")

            # 19 维特征网格
            features = result.get('features')
            _render_feature_telemetry_grid(features)

            # 模型推断控制台
            _render_model_inference_block(features, source_key='upload')
        else:
            st.error(f"❌ 图像视觉分割失败: {result.get('error', '无法有效提取鸡蛋轮廓')}")
            st.info("💡 拍摄建议：确保背景与蛋体有明显明度或色差对比，避免强烈高光反光。")


def _render_quick_select_workflow():
    """90枚标本库快速诊断工作流"""
    st.markdown("""
    <div style="background: rgba(13, 19, 33, 0.7); border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 700; color: #00E5FF; font-size: 0.98rem; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.45rem;">
            <span>🎯</span><span>标准化标本库即时调用与模型验证 (Specimen Telemetry & AI Diagnostic)</span>
        </div>
        <div style="color: #94A3B8; font-size: 0.86rem; line-height: 1.5;">
            从标准化几何标本库中挑选样本，直接加载其高精度光学轮廓图谱并调用决策模型执行动力学稳定性分类验证。
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        # 扫描现有可用鸡蛋图像
        images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'egg_images')
        available_eggs = list(range(1, 91))
        if os.path.exists(images_dir):
            found_eggs = [
                int(f.split('号')[0]) for f in os.listdir(images_dir)
                if f.endswith('.jpg') and '号' in f and f.split('号')[0].isdigit()
            ]
            if found_eggs:
                available_eggs = sorted(list(set(found_eggs)))

        egg_id = st.selectbox(
            "选择待诊断标本编号",
            available_eggs,
            format_func=lambda x: f"第 {x} 号鸡蛋标本",
            key="quick_egg_select_picker"
        )

        image_path = get_egg_image_path(egg_id)

        fusion_df = load_fusion_data()
        r_lbl, r_clr, r_emo = '未知', '#94A3B8', '⚪'
        if fusion_df is not None:
            id_col = 'egg_id' if 'egg_id' in fusion_df.columns else ('EggID' if 'EggID' in fusion_df.columns else None)
            rsi_col = 'RSI' if 'RSI' in fusion_df.columns else ('RSI_GroupNum' if 'RSI_GroupNum' in fusion_df.columns else None)
            if id_col and rsi_col:
                matched = fusion_df[fusion_df[id_col] == egg_id]
                if not matched.empty:
                    true_risk = int(matched.iloc[0].get(rsi_col, 1))
                    r_lbl, r_clr, r_emo = RSI_LABELS.get(true_risk, ('未知', '#94A3B8', '⚪'))

        # 全息 HUD 扫描观测舱包装
        st.markdown(f"""
        <div class="hud-chamber" style="border-top: 3px solid {r_clr}; margin-top: 0.4rem;">
            <div class="hud-bracket hud-bracket-tl"></div>
            <div class="hud-bracket hud-bracket-tr"></div>
            <div class="hud-bracket hud-bracket-bl"></div>
            <div class="hud-bracket hud-bracket-br"></div>
            <div class="hud-laser-scanner"></div>

            <div class="hud-telemetry-header">
                <div>
                    <span class="hud-tag">SPECIMEN #{egg_id:02d}</span>
                    <span style="margin-left: 0.35rem; color: #E2E8F0; font-weight: 600;">ACTIVE TARGET</span>
                </div>
                <div style="background: {r_clr}22; border: 1px solid {r_clr}; color: {r_clr};
                            padding: 0.12rem 0.55rem; border-radius: 9999px; font-weight: 700; font-size: 0.72rem;">
                    {r_emo} 实测: {r_lbl}
                </div>
            </div>
        """, unsafe_allow_html=True)

        if os.path.exists(image_path):
            st.image(image_path, caption=f"标本 {egg_id} 号 — 边缘拟合几何轮廓", use_container_width=True)
        else:
            st.warning(f"标本图像载入中: {egg_id}号")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        with st.spinner("正在提取形态学参数..."):
            features = extract_features_from_image(image_path) if os.path.exists(image_path) else None
        _render_feature_telemetry_grid(features)

    # 模型推断
    _render_model_inference_block(features, source_key='quick')


def _render_feature_telemetry_grid(features):
    """渲染 19 维特征高级 Bento 网格"""
    if not features:
        st.info("特征提取中...")
        return

    st.markdown("""
    <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.45rem;">
        <span>📐</span><span>实时形态解耦特征 (19-Dimension Telemetry Bento)</span>
    </div>
    """, unsafe_allow_html=True)

    # 4 大核心关键几何指标
    core_items = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数(ESI)', '#00E5FF', '饱满度指标'),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数', '#A78BFA', '曲率偏心差'),
        ('Static_Eccentricity_离心率', '离心率', '#38BDF8', '焦点比值'),
        ('Static_Circularity_圆形度', '圆形度', '#10B981', '圆滑规整度'),
    ]

    c4 = st.columns(4)
    for idx, (col_k, label, accent_c, sub) in enumerate(core_items):
        v = features.get(col_k, 0)
        v_num = float(v) if isinstance(v, (int, float)) else 0.0
        with c4[idx]:
            st.markdown(f"""
            <div style="background: rgba(13, 19, 33, 0.85); border: 1px solid rgba(255,255,255,0.08); border-top: 2.5px solid {accent_c};
                        border-radius: 12px; padding: 0.65rem 0.8rem; margin-bottom: 0.6rem;">
                <div style="font-size: 0.72rem; color: #94A3B8;">{label}</div>
                <div style="font-family: var(--font-mono); font-size: 1.18rem; font-weight: 800; color: #FFFFFF; margin: 0.15rem 0;">
                    {v_num:.4f}
                </div>
                <div style="font-size: 0.65rem; color: {accent_c}; font-weight: 600;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # 8 项空间尺寸指标
    geom_items = [
        ('Static_Area_像素面积', '像素面积', 'px²'),
        ('Static_Perimeter_轮廓周长', '周长', 'px'),
        ('Static_MajorAxisLength_长轴像素长度', '长轴', 'px'),
        ('Static_MinorAxisLength_短轴像素长度', '短轴', 'px'),
        ('Static_Solidity_坚实度', '坚实度', ''),
        ('Static_Extent_延展度', '延展度', ''),
        ('Static_EquivalentDiameter_等效圆直径', '等效直径', 'px'),
        ('Static_MajorAxisOffsetRatio_长轴偏移率', '主轴偏移', ''),
    ]

    c_cols = st.columns(4)
    for idx, (col_k, label, unit) in enumerate(geom_items):
        v = features.get(col_k, 0)
        with c_cols[idx % 4]:
            if isinstance(v, float):
                v_str = f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            else:
                v_str = str(v)
            u_str = f" <span style='font-size:0.65rem;color:#64748B;'>{unit}</span>" if unit else ""
            st.markdown(f"""
            <div class="metric-cell" style="padding: 0.55rem 0.75rem; margin-bottom: 0.45rem;">
                <div class="metric-cell-value" style="font-size: 1.05rem;">{v_str}{u_str}</div>
                <div class="metric-cell-label" style="font-size: 0.7rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # 7 个 Hu 矩紧凑显示
    hu_cols = st.columns(7)
    for i in range(1, 8):
        h_val = features.get(f'Static_Hu{i}', 0)
        with hu_cols[i - 1]:
            h_disp = f"{h_val:.2e}" if isinstance(h_val, float) else str(h_val)
            st.markdown(f"""
            <div style="background: rgba(11, 16, 29, 0.75); border: 1px solid rgba(167, 139, 250, 0.18);
                        border-radius: 8px; padding: 0.4rem 0.25rem; text-align: center;">
                <div style="font-size: 0.65rem; color: #C084FC; font-weight: 700; font-family: var(--font-mono);">Hu{i}</div>
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #E2E8F0;">{h_disp}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_model_inference_block(features, source_key='default'):
    """模型选择与推断执行卡片"""
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⚡</span>
        <span>智能决策引擎与推断执行 (Inference & Sorting Execution)</span>
    </div>
    """, unsafe_allow_html=True)

    # 模型卡片横幅比武
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin-bottom: 1rem;">
        <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.35); border-radius: 12px; padding: 0.65rem 0.9rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #00E5FF; font-weight: 700;">
                <span>SVM 支持向量机</span><span>🏆 推荐最优</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 0.2rem;">
                AUC: 85.85% <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 400;">/ Acc: 80.0%</span>
            </div>
        </div>
        <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 0.65rem 0.9rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #C084FC; font-weight: 700;">
                <span>随机森林 (RF)</span><span>高鲁棒性</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 0.2rem;">
                AUC: 84.14% <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 400;">/ Acc: 82.2%</span>
            </div>
        </div>
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 0.65rem 0.9rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #F59E0B; font-weight: 700;">
                <span>梯度提升 (GBDT)</span><span>残差加权</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 0.2rem;">
                AUC: 81.33% <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 400;">/ Acc: 80.0%</span>
            </div>
        </div>
        <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 0.65rem 0.9rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94A3B8; font-weight: 700;">
                <span>逻辑回归 (LR)</span><span>线性基准</span>
            </div>
            <div style="font-family: var(--font-mono); font-size: 1.1rem; font-weight: 800; color: #FFFFFF; margin-top: 0.2rem;">
                AUC: 79.52% <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 400;">/ Acc: 76.7%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_m, col_btn = st.columns([1.4, 1.2])

    with col_m:
        model_keys = list(MODEL_NAMES.keys())
        def_idx = model_keys.index('svm') if 'svm' in model_keys else 0
        selected_model = st.selectbox(
            "选择推断决策模型",
            model_keys,
            index=def_idx,
            format_func=lambda x: f"{MODEL_NAMES.get(x, x)} {'(推荐最优核心)' if x=='svm' else ''}",
            key=f"{source_key}_model_picker"
        )

    with col_btn:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 启动多模态推断与产线分选评估", type="primary", use_container_width=True, key=f"{source_key}_run_btn")

    session_pred_key = f"{source_key}_pred_result"

    if run_btn:
        if features:
            with st.spinner("🧠 机器学习模型推理计算中..."):
                pred, probs = predict_risk(features, selected_model)
            if pred is not None:
                st.session_state[session_pred_key] = (pred, probs, selected_model)
            else:
                st.error("❌ 模型预测失败，请核查模型权重文件。")
        else:
            st.error("⚠️ 未检测到有效特征数据，无法启动预测。")

    if session_pred_key in st.session_state:
        pred, probs, m_name = st.session_state[session_pred_key]
        _display_diagnostic_report(pred, probs, m_name)


def _display_diagnostic_report(prediction, probabilities, model_name):
    """展示顶级科技感自动化分选产线决策报告"""
    risk_name, risk_color, risk_icon = RSI_LABELS.get(prediction, ('未知', '#94A3B8', '⚪'))

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.5rem;">
        <span>📋</span><span>自动化分选决策与健康诊断报告 (Sorting & Actuator Report)</span>
    </div>
    """, unsafe_allow_html=True)

    card_col, chart_col = st.columns([1.1, 1.5])

    with card_col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color}22 0%, rgba(13, 19, 33, 0.95) 100%);
                    border: 2px solid {risk_color};
                    border-radius: 18px;
                    padding: 1.6rem 1.4rem;
                    text-align: center;
                    box-shadow: 0 0 32px {risk_color}38, inset 0 1px 0 rgba(255,255,255,0.15);
                    position: relative;
                    overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 80px; height: 80px;
                        background: {risk_color}33; border-radius: 50%; filter: blur(20px);"></div>
            <div style="font-size: 3.2rem; margin-bottom: 0.4rem; filter: drop-shadow(0 0 14px {risk_color});">{risk_icon}</div>
            <div style="font-family: var(--font-display); font-size: 2rem; font-weight: 800; color: {risk_color}; margin-bottom: 0.2rem; letter-spacing: -0.01em;">
                {risk_name}
            </div>
            <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 0.9rem;">
                滚落稳定性风险指数: <b style="color: #FFFFFF; font-family: var(--font-mono);">Level {prediction}</b>
            </div>
            <div style="display: inline-block; background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
                        padding: 0.35rem 0.9rem; border-radius: 9999px; font-size: 0.76rem; color: #CBD5E1; font-family: var(--font-mono);">
                推断引擎: {MODEL_NAMES.get(model_name, model_name)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        # 置信度分布水平条形图
        p_low = probabilities[0] if len(probabilities) > 0 else 0
        p_mid = probabilities[1] if len(probabilities) > 1 else 0
        p_high = probabilities[2] if len(probabilities) > 2 else 0

        prob_data = [
            ('低风险 🟢', p_low, '#10B981'),
            ('中风险 🟡', p_mid, '#F59E0B'),
            ('高风险 🔴', p_high, '#EF4444'),
        ]

        if HAS_PLOTLY:
            fig = go.Figure()
            for label, prob, bar_c in prob_data:
                fig.add_trace(go.Bar(
                    y=[label],
                    x=[prob],
                    orientation='h',
                    marker=dict(color=bar_c, cornerradius=6),
                    text=[f"{prob:.1%}"],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='#FFFFFF', size=13, family='JetBrains Mono, monospace'),
                    hoverinfo='none',
                    showlegend=False
                ))

            fig.update_layout(
                title=dict(text='各风险等级后验概率分布 (Posterior Probability)', font=dict(color='#E2E8F0', size=13)),
                xaxis=dict(range=[0, 1], tickformat='.0%', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.06)'),
                yaxis=dict(tickfont=dict(color='#E2E8F0', size=12), categoryorder='array', categoryarray=['高风险 🔴', '中风险 🟡', '低风险 🟢']),
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=20, t=35, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 纯 HTML/CSS 科技水平进度条优雅降级
            st.markdown(f"""
            <div style="background: rgba(13, 19, 33, 0.75); border: 1px solid rgba(255,255,255,0.08);
                        border-radius: 14px; padding: 1.2rem; height: 100%;">
                <div style="font-size: 0.9rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.9rem;">
                    各风险等级后验概率分布 (Posterior Probability)
                </div>
                <div style="margin-bottom: 0.7rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.25rem;">
                        <span>低风险 🟢</span><span style="font-family: var(--font-mono); color: #10B981; font-weight: 700;">{p_low:.1%}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.06); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: #10B981; width: {max(2, int(p_low*100))}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 0.7rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.25rem;">
                        <span>中风险 🟡</span><span style="font-family: var(--font-mono); color: #F59E0B; font-weight: 700;">{p_mid:.1%}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.06); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: #F59E0B; width: {max(2, int(p_mid*100))}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94A3B8; margin-bottom: 0.25rem;">
                        <span>高风险 🔴</span><span style="font-family: var(--font-mono); color: #EF4444; font-weight: 700;">{p_high:.1%}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.06); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: #EF4444; width: {max(2, int(p_high*100))}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 工业产线分选决策建议指示
    advice_registry = {
        1: {
            'title': '✅ A 级产线决策：允许高速自动化传输分级',
            'action': '【快速通过】进入高通量自动包装线（线速可达 1.2 m/s），采用标准滚道输送。',
            'mechanism': '该蛋体几何对称性高，长轴偏移率极低，滚落时自转力矩均衡，运动轨迹直线平稳，无磕碰破壳风险。',
            'color': '#10B981',
            'icon': '🟢'
        },
        2: {
            'title': '⚠️ B 级产线决策：降速缓冲或柔性气动调姿',
            'action': '【减速缓行】建议降低输送带流速至 0.6 m/s 以下，或启用侧边软质弹性挡板防止侧滑。',
            'mechanism': '检测到存在轻度各向异性或长轴偏心，在倾角滚落中会产生周期性摆动与姿态倾侧，需防范堆叠挤压。',
            'color': '#F59E0B',
            'icon': '🟡'
        },
        3: {
            'title': '🚨 C 级产线决策：高风险易损预警，进入专用柔性通道',
            'action': '【剔除/人工复核】气动推杆移出主高速线，转入低速柔性海绵通道单独装箱，避免密集碰撞。',
            'mechanism': '该蛋体长短轴曲率突变显著，不对称指数偏大，滚落过程中极易失稳倾覆、发生剧烈偏航或与邻近蛋品剧烈撞击破损。',
            'color': '#EF4444',
            'icon': '🔴'
        }
    }

    ad = advice_registry.get(prediction, advice_registry[1])
    st.markdown(f"""
    <div style="background: rgba(13, 19, 33, 0.9); border-left: 5px solid {ad['color']}; border-radius: 0 14px 14px 0;
                padding: 1.2rem 1.5rem; margin-top: 1.1rem; border-top: 1px solid rgba(255,255,255,0.06);
                border-bottom: 1px solid rgba(255,255,255,0.06); border-right: 1px solid rgba(255,255,255,0.06);
                box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
        <div style="font-weight: 700; color: {ad['color']}; font-size: 1.08rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>{ad['icon']}</span><span>{ad['title']}</span>
        </div>
        <div style="color: #F1F5F9; font-size: 0.92rem; margin-bottom: 0.4rem; line-height: 1.55;">
            <b>产线执行动作：</b> {ad['action']}
        </div>
        <div style="color: #94A3B8; font-size: 0.86rem; line-height: 1.55;">
            <b>机理成因分析：</b> {ad['mechanism']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_research_insights():
    """论文配套图表与机理验证视窗"""
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    with st.expander("📊 查看论文配套图表：3D 聚类分布与滚落轨迹追踪图谱 (Clustering & Trajectory)", expanded=False):
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
        c1, c2 = st.columns(2)
        with c1:
            rsi_3d = os.path.join(assets_dir, 'rsi_3d_cluster.jpg')
            if os.path.exists(rsi_3d):
                st.image(rsi_3d, caption="图 3.1 动态易损性指数 (RSI) 在三维特征空间的聚类映射图", use_container_width=True)
        with c2:
            traj_img = os.path.join(assets_dir, 'trajectory_tracking.jpg')
            if os.path.exists(traj_img):
                st.image(traj_img, caption="图 3.2 动态滚落过程高速相机实时轨迹追踪效果", use_container_width=True)
        st.caption("研究数据源自物理滚落试验台 270 组连续追踪实测记录。")
