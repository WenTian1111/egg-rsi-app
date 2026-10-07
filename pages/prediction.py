"""
预测页面 - 鸡蛋滚落风险智能预测与分选决策
西南大学大学生创新创业训练计划项目 (S202510635378)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

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
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0 0.6rem;">
                <span style="font-weight: 600; color: #FFFFFF; font-size: 0.95rem;">📷 视觉分割流水线中间态观测</span>
                <span style="background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0, 229, 255, 0.3);
                             color: #00E5FF; padding: 0.15rem 0.6rem; border-radius: 6px; font-size: 0.75rem;">
                    策略引擎: {strategy}
                </span>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(4)
            step_defs = [
                ('original', '01. 原始图像'),
                ('grayscale', '02. 灰度矩阵'),
                ('hsv_mask', '03. 分割掩膜'),
                ('contour_viz', '04. 拟合轮廓'),
            ]
            for idx, (key, label) in enumerate(step_defs):
                with cols[idx]:
                    img = result['steps'].get(key)
                    if img is not None:
                        st.image(img, caption=label, use_container_width=True)
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
    <div style="background: rgba(14, 20, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;">
        <div style="font-weight: 600; color: #00E5FF; font-size: 0.95rem; margin-bottom: 0.3rem;">
            🎯 标准化标本库即时调用与模型验证
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem;">
            从国家级大创实验收集的 90 枚标准化几何标本中挑选样本，直接加载其高精度轮廓并调用模型执行稳定性分类验证。
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

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
            "选择标本编号",
            available_eggs,
            format_func=lambda x: f"第 {x} 号鸡蛋标本",
            key="quick_egg_select_picker"
        )

        image_path = get_egg_image_path(egg_id)
        if os.path.exists(image_path):
            st.image(image_path, caption=f"标本 {egg_id} 号 — 几何边缘提取图", use_container_width=True)
        else:
            st.warning(f"标本图像载入中: {egg_id}号")

        fusion_df = load_fusion_data()
        if fusion_df is not None:
            id_col = 'egg_id' if 'egg_id' in fusion_df.columns else ('EggID' if 'EggID' in fusion_df.columns else None)
            rsi_col = 'RSI' if 'RSI' in fusion_df.columns else ('RSI_GroupNum' if 'RSI_GroupNum' in fusion_df.columns else None)
            if id_col and rsi_col:
                matched = fusion_df[fusion_df[id_col] == egg_id]
                if not matched.empty:
                    true_risk = int(matched.iloc[0].get(rsi_col, 1))
                    r_lbl, r_clr, r_emo = RSI_LABELS.get(true_risk, ('未知', '#94A3B8', '⚪'))
                    st.markdown(f"""
                    <div style="background: {r_clr}18; border: 1px solid {r_clr}66; border-radius: 10px;
                                padding: 0.6rem; text-align: center; margin-top: 0.6rem;">
                        <span style="font-size: 1.1rem;">{r_emo}</span>
                        <span style="font-size: 0.82rem; color: #94A3B8; margin-left: 0.3rem;">物理实验实测标签:</span>
                        <span style="font-weight: 700; color: {r_clr}; margin-left: 0.3rem;">{r_lbl}</span>
                    </div>
                    """, unsafe_allow_html=True)

    with col_right:
        with st.spinner("正在提取形态学参数..."):
            features = extract_features_from_image(image_path) if os.path.exists(image_path) else None
        _render_feature_telemetry_grid(features)

    # 模型推断
    _render_model_inference_block(features, source_key='quick')


def _render_feature_telemetry_grid(features):
    """渲染 19 维特征紧凑网格"""
    if not features:
        st.info("特征提取中...")
        return

    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 600; color: #FFFFFF; margin: 1rem 0 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>📐</span><span>实时形态解耦特征 (19-Dimension Telemetry)</span>
    </div>
    """, unsafe_allow_html=True)

    geom_items = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数(ESI)', ''),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数', ''),
        ('Static_Eccentricity_离心率', '离心率', ''),
        ('Static_Area_像素面积', '像素面积', 'px²'),
        ('Static_Perimeter_轮廓周长', '周长', 'px'),
        ('Static_MajorAxisLength_长轴像素长度', '长轴', 'px'),
        ('Static_MinorAxisLength_短轴像素长度', '短轴', 'px'),
        ('Static_Circularity_圆形度', '圆形度', ''),
        ('Static_Solidity_坚实度', '坚实度', ''),
        ('Static_Extent_延展度', '延展度', ''),
        ('Static_EquivalentDiameter_等效圆直径', '等效直径', 'px'),
        ('Static_MajorAxisOffsetRatio_长轴偏移率', '长轴偏移率', ''),
    ]

    c_cols = st.columns(6)
    for idx, (col_k, label, unit) in enumerate(geom_items):
        v = features.get(col_k, 0)
        with c_cols[idx % 6]:
            if isinstance(v, float):
                v_str = f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
            else:
                v_str = str(v)
            u_str = f" <span style='font-size:0.6rem;color:#64748B;'>{unit}</span>" if unit else ""
            st.markdown(f"""
            <div style="background: rgba(14, 20, 35, 0.7); border: 1px solid rgba(255,255,255,0.06);
                        border-radius: 8px; padding: 0.45rem 0.55rem; text-align: center; margin-bottom: 0.4rem;">
                <div style="font-family: var(--font-mono); font-size: 0.92rem; font-weight: 700; color: #00E5FF;">{v_str}{u_str}</div>
                <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 0.15rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # 7 个 Hu 矩紧凑显示
    hu_cols = st.columns(7)
    for i in range(1, 8):
        h_val = features.get(f'Static_Hu{i}', 0)
        with hu_cols[i - 1]:
            h_disp = f"{h_val:.2e}" if isinstance(h_val, float) else str(h_val)
            st.markdown(f"""
            <div style="background: rgba(10, 15, 26, 0.6); border: 1px solid rgba(255,255,255,0.04);
                        border-radius: 6px; padding: 0.35rem 0.2rem; text-align: center;">
                <div style="font-size: 0.65rem; color: #A78BFA; font-weight: 600;">Hu{i}</div>
                <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #CBD5E1;">{h_disp}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_model_inference_block(features, source_key='default'):
    """模型选择与推断执行卡片"""
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="section-title-icon">⚡</span>
        <span>智能决策引擎与推断执行 (Inference & Sorting Execution)</span>
    </div>
    """, unsafe_allow_html=True)

    col_m, col_btn = st.columns([1.5, 1])

    with col_m:
        model_keys = list(MODEL_NAMES.keys())
        def_idx = model_keys.index('svm') if 'svm' in model_keys else 0
        selected_model = st.selectbox(
            "选择分类预测模型",
            model_keys,
            index=def_idx,
            format_func=lambda x: f"{MODEL_NAMES.get(x, x)} {'(推荐最优)' if x=='svm' else ''}",
            key=f"{source_key}_model_picker"
        )

    with col_btn:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 启动智能推断与产线分选评估", type="primary", use_container_width=True, key=f"{source_key}_run_btn")

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

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
        <span>📋</span><span>自动化分选决策与健康诊断报告 (Sorting & Actuator Report)</span>
    </div>
    """, unsafe_allow_html=True)

    card_col, chart_col = st.columns([1.1, 1.4])

    with card_col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color}18 0%, rgba(14, 20, 35, 0.9) 100%);
                    border: 2px solid {risk_color};
                    border-radius: 16px;
                    padding: 1.5rem 1.2rem;
                    text-align: center;
                    box-shadow: 0 0 25px {risk_color}33;
                    position: relative;
                    overflow: hidden;">
            <div style="position: absolute; top: -15px; right: -15px; width: 60px; height: 60px;
                        background: {risk_color}22; border-radius: 50%; filter: blur(15px);"></div>
            <div style="font-size: 3rem; margin-bottom: 0.4rem; filter: drop-shadow(0 0 10px {risk_color});">{risk_icon}</div>
            <div style="font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; color: {risk_color}; margin-bottom: 0.2rem;">
                {risk_name}
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 0.8rem;">
                动态滚落易损性指数等级: <b style="color: #FFFFFF;">Level {prediction}</b>
            </div>
            <div style="display: inline-block; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.08);
                        padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.75rem; color: #94A3B8;">
                推断模型核心: {MODEL_NAMES.get(model_name, model_name)}
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

        fig = go.Figure()
        for label, prob, bar_c in prob_data:
            fig.add_trace(go.Bar(
                y=[label],
                x=[prob],
                orientation='h',
                marker=dict(color=bar_c, cornerradius=5),
                text=[f"{prob:.1%}"],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='#FFFFFF', size=13, family='JetBrains Mono, monospace'),
                hoverinfo='none',
                showlegend=False
            ))

        fig.update_layout(
            title=dict(text='各风险类别后验概率分布 (Class Probability Distribution)', font=dict(color='#E2E8F0', size=13)),
            xaxis=dict(range=[0, 1], tickformat='.0%', tickfont=dict(color='#64748B'), gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(tickfont=dict(color='#E2E8F0', size=11), categoryorder='array', categoryarray=['高风险 🔴', '中风险 🟡', '低风险 🟢']),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=20, t=35, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

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
    <div style="background: rgba(14, 20, 35, 0.85); border-left: 5px solid {ad['color']}; border-radius: 0 12px 12px 0;
                padding: 1.1rem 1.4rem; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.05);
                border-bottom: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05);">
        <div style="font-weight: 700; color: {ad['color']}; font-size: 1.05rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>{ad['icon']}</span><span>{ad['title']}</span>
        </div>
        <div style="color: #F1F5F9; font-size: 0.9rem; margin-bottom: 0.35rem; line-height: 1.5;">
            <b>产线执行动作：</b> {ad['action']}
        </div>
        <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
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
        st.caption("研究数据源自西南大学物理滚落试验台270组连续追踪实测记录。")
