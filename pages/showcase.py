"""展示页面 - 图像处理流水线 + 数据库浏览"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from PIL import Image

from utils.data_loader import (
    load_fusion_data, load_model_metrics, load_feature_importance,
    RSI_LABELS, STATIC_FEATURES_19, get_egg_image_path, get_processing_images
)


def show_showcase():
    st.markdown('<div class="section-header">🔬 展示</div>', unsafe_allow_html=True)

    try:
        df = load_fusion_data()
        if df is None:
            st.error("❌ 数据加载失败，请检查数据文件是否存在")
            return
    except Exception as e:
        st.error(f"❌ 数据加载失败: {str(e)}")
        return

    if 'selected_egg' not in st.session_state:
        st.session_state.selected_egg = sorted(df["EggID"].unique())[0]

    tab1, tab2 = st.tabs(["🧪 图像处理流水线", "📋 数据库浏览"])

    with tab1:
        _show_pipeline_tab(df)

    with tab2:
        _show_browser_tab(df)


def _show_pipeline_tab(df):
    st.markdown("### 📷 图像处理流程（1号鸡蛋）")

    try:
        img_paths = get_processing_images()
    except Exception as e:
        st.warning(f"无法加载处理图像: {str(e)}")
        img_paths = {}

    image_names = {
        'original': '原图',
        'grayscale': '灰度图',
        'mask': '二值掩膜图',
        'contour': '轮廓质心包围盒图'
    }

    row1_cols = st.columns(2)
    row2_cols = st.columns(2)

    for idx, (key, caption) in enumerate(image_names.items()):
        img_path = img_paths.get(key)
        col = row1_cols[idx] if idx < 2 else row2_cols[idx - 2]

        with col:
            if img_path and os.path.exists(img_path):
                try:
                    st.image(img_path, caption=caption, use_container_width=False, width=350)
                except Exception:
                    st.markdown(f'''
                    <div style="width: 350px; height: 280px; background-color: #1A1C23; border: 1px solid #2D2D3D;
                                border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                        <span style="color: #ADB5BD;">图像加载失败</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div style="width: 350px; height: 280px; background-color: #1A1C23; border: 1px solid #2D2D3D;
                            border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #ADB5BD;">暂无图像</span>
                </div>
                ''', unsafe_allow_html=True)

    st.markdown("### 📊 特征提取结果")

    egg_ids = sorted(df["EggID"].unique())
    selected_egg = st.selectbox(
        "选择鸡蛋",
        egg_ids,
        index=egg_ids.index(st.session_state.selected_egg) if st.session_state.selected_egg in egg_ids else 0,
        format_func=lambda x: f"{x}号鸡蛋"
    )
    st.session_state.selected_egg = selected_egg

    egg_row = df[df['EggID'] == selected_egg].iloc[0]
    rsi_group = int(egg_row['RSI_GroupNum'])

    feat_cols = st.columns(3)
    features = [
        ('Static_ShapeIndex_机器视觉ESI', '蛋形指数 (ESI)'),
        ('Static_AsymmetryIndex_不对称指数', '不对称指数'),
        ('Static_Eccentricity_离心率', '离心率'),
        ('Static_Area_像素面积', '像素面积'),
        ('Static_Perimeter_轮廓周长', '轮廓周长'),
        ('Static_Circularity_圆形度', '圆形度'),
    ]

    for idx, (col_name, label) in enumerate(features):
        with feat_cols[idx % 3]:
            try:
                val = egg_row.get(col_name, 0)
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{val:.4f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                ''', unsafe_allow_html=True)
            except Exception:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">{label}</div>
                </div>
                ''', unsafe_allow_html=True)

    if idx % 3 != 0:
        for _ in range(3 - (idx % 3)):
            st.empty()

    st.markdown("#### 🔬 Hu 矩特征")

    hu_features = ['Static_Hu1', 'Static_Hu2', 'Static_Hu3', 'Static_Hu4',
                    'Static_Hu5', 'Static_Hu6', 'Static_Hu7']
    hu_labels = ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7']

    hu_data = []
    for hu_col in hu_features:
        try:
            val = egg_row.get(hu_col, 0)
            hu_data.append({"Hu矩": hu_col.replace('Static_', ''), "数值": val})
        except Exception:
            hu_data.append({"Hu矩": hu_col.replace('Static_', ''), "数值": 0})

    try:
        st.table(pd.DataFrame(hu_data))
    except Exception:
        st.info("Hu矩数据暂不可用")

    st.markdown("### 🎯 风险预测")

    risk_col1, risk_col2 = st.columns([1, 1])

    with risk_col1:
        risk_label, risk_color, risk_icon = RSI_LABELS.get(rsi_group, ('未知', '#ADB5BD', '⚪'))
        st.markdown(f'''
        <div class="card" style="border-left: 4px solid {risk_color};">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{risk_icon} 风险等级</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {risk_color};">{risk_label}</div>
        </div>
        ''', unsafe_allow_html=True)

    with risk_col2:
        advice = {
            1: "该鸡蛋滚落稳定性良好，可采用标准输送流程，无需特殊处理。",
            2: "该鸡蛋滚落时存在一定偏移与姿态波动风险，建议降低输送速度或增加缓冲装置。",
            3: "该鸡蛋滚落稳定性较差，轨迹偏移显著，建议单独通道处理，避免与其它鸡蛋碰撞。"
        }.get(rsi_group, "暂无建议")
        st.markdown(f'''
        <div class="card">
            <div style="font-size: 1.1rem; font-weight: 600; color: #00B4D8; margin-bottom: 0.5rem;">📋 分选建议</div>
            <div style="color: #E0E0E0;">{advice}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("### 📈 RSI 风险分布")

    try:
        risk_counts = df['RSI_GroupNum'].value_counts().sort_index()
        total_count = len(df)

        bar_data = []
        for grp in [1, 2, 3]:
            label, color, _ = RSI_LABELS.get(grp, ('未知', '#ADB5BD', ''))
            count = risk_counts.get(grp, 0)
            pct = count / total_count * 100 if total_count > 0 else 0
            bar_data.append({"风险等级": label, "数量": count, "占比": pct, "颜色": color})

        fig_bar = go.Figure()
        for item in bar_data:
            fig_bar.add_trace(go.Bar(
                x=[item["风险等级"]],
                y=[item["数量"]],
                marker_color=item["颜色"],
                text=f"{item['占比']:.1f}%",
                textposition='auto',
                name=item["风险等级"]
            ))
        fig_bar.update_layout(
            title=dict(text='鸡蛋风险等级分布', font=dict(color='#ADB5BD', size=14)),
            xaxis=dict(title='风险等级', tickfont=dict(color='#ADB5BD'), gridcolor='#2D2D3D'),
            yaxis=dict(title='鸡蛋数量', tickfont=dict(color='#ADB5BD'), gridcolor='#2D2D3D'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        st.info("风险分布图表暂不可用")


def _show_browser_tab(df):
    with st.sidebar:
        st.markdown("### 🔍 筛选条件")
        risk_options = {"低风险 (1)": 1, "中风险 (2)": 2, "高风险 (3)": 3}
        selected_risks = st.multiselect(
            "按风险等级筛选",
            list(risk_options.keys()),
            default=list(risk_options.keys())
        )
        selected_nums = [risk_options[r] for r in selected_risks]
        try:
            filtered = df[df['RSI_GroupNum'].isin(selected_nums)]
        except Exception:
            filtered = df

        egg_ids = sorted(filtered['EggID'].unique())
        selected_egg = st.selectbox("选择鸡蛋", egg_ids, format_func=lambda x: f"{x}号鸡蛋")

        st.markdown("---")
        try:
            total_eggs = df['EggID'].nunique()
            st.metric("总鸡蛋数", total_eggs)
            st.metric("总试验次数", len(df))
        except Exception:
            pass

    egg_data = df[df['EggID'] == selected_egg]
    if len(egg_data) == 0:
        st.warning("暂无该鸡蛋的数据")
        return

    egg_row = egg_data.iloc[0]
    rsi_group = int(egg_row['RSI_GroupNum'])

    col_img, col_info = st.columns([1, 1])

    with col_img:
        try:
            img_path = get_egg_image_path(selected_egg)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=f"{selected_egg}号鸡蛋 — 轮廓提取结果", use_container_width=True)
            else:
                st.markdown(f'''
                <div style="background-color: #1A1C23; border: 1px solid #2D2D3D; border-radius: 12px;
                            padding: 3rem; text-align: center;">
                    <span style="color: #ADB5BD;">暂无图像</span>
                </div>
                ''', unsafe_allow_html=True)
        except Exception:
            st.markdown(f'''
            <div style="background-color: #1A1C23; border: 1px solid #2D2D3D; border-radius: 12px;
                        padding: 3rem; text-align: center;">
                <span style="color: #ADB5BD;">暂无图像</span>
            </div>
            ''', unsafe_allow_html=True)

    with col_info:
        try:
            esi_val = egg_row.get('Static_ShapeIndex_机器视觉ESI', 0)
            asym_val = egg_row.get('Static_AsymmetryIndex_不对称指数', 0)
            ecc_val = egg_row.get('Static_Eccentricity_离心率', 0)
            circ_val = egg_row.get('Static_Circularity_圆形度', 0)
        except Exception:
            esi_val = asym_val = ecc_val = circ_val = 0

        risk_color_map = {1: '#4ECDC4', 2: '#FFE66D', 3: '#FF6B6B'}
        risk_label_map = {1: '低风险', 2: '中风险', 3: '高风险'}
        risk_color = risk_color_map.get(rsi_group, '#ADB5BD')
        risk_label = risk_label_map.get(rsi_group, '未知')

        st.markdown(f'''
        <div class="card">
            <div style="color: #00B4D8; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">{selected_egg}号鸡蛋</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;">
                <div><span style="color: #ADB5BD;">试验次数:</span><br><span style="color: white; font-size: 1.2rem;">{len(egg_data)} 次</span></div>
                <div><span style="color: #ADB5BD;">风险等级:</span><br><span style="color: {risk_color}; font-size: 1.2rem;">{risk_label}</span></div>
                <div><span style="color: #ADB5BD;">蛋形指数 (ESI):</span><br><span style="color: white; font-size: 1.2rem;">{esi_val:.4f}</span></div>
                <div><span style="color: #ADB5BD;">不对称指数:</span><br><span style="color: white; font-size: 1.2rem;">{asym_val:.4f}</span></div>
                <div><span style="color: #ADB5BD;">离心率:</span><br><span style="color: white; font-size: 1.2rem;">{ecc_val:.4f}</span></div>
                <div><span style="color: #ADB5BD;">圆形度:</span><br><span style="color: white; font-size: 1.2rem;">{circ_val:.4f}</span></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    feat_tab, model_tab = st.tabs(["📊 特征雷达图", "📉 模型对比"])

    with feat_tab:
        features = ['Static_ShapeIndex_机器视觉ESI', 'Static_AsymmetryIndex_不对称指数',
                    'Static_Eccentricity_离心率', 'Static_Circularity_圆形度',
                    'Static_Solidity_坚实度', 'Static_Extent_延展度']
        labels_cn = ['蛋形指数', '不对称指数', '离心率', '圆形度', '坚实度', '延展度']

        try:
            norm_vals = []
            for col in features:
                col_min = df[col].min()
                col_max = df[col].max()
                val = egg_row[col]
                norm_val = (val - col_min) / (col_max - col_min) if col_max > col_min else 0.5
                norm_vals.append(norm_val)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=labels_cn + [labels_cn[0]],
                fill='toself',
                name=f'{selected_egg}号鸡蛋',
                line=dict(color='#00B4D8', width=2),
                fillcolor='rgba(0, 180, 216, 0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#1A1C23',
                    radialaxis=dict(visible=True, range=[0, 1], color='#ADB5BD',
                                    gridcolor='#2D2D3D'),
                    angularaxis=dict(color='#ADB5BD', gridcolor='#2D2D3D')
                ),
                showlegend=True,
                legend=dict(font=dict(color='white')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=80, r=80, t=30, b=30),
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception:
            st.error("雷达图渲染失败，请检查数据完整性")

        with st.expander("查看详细特征数据"):
            try:
                feat_data = {label: egg_row[col] for col, label in zip(features, labels_cn)}
                st.table(pd.DataFrame([feat_data]))
            except Exception:
                st.info("特征数据暂不可用")

    with model_tab:
        try:
            metrics_df = load_model_metrics()
            if metrics_df is not None and 'ModelName' in metrics_df.columns:
                metrics_df = metrics_df.dropna(subset=['ModelName'])
                metrics_df = metrics_df[metrics_df['ModelName'].str.strip() != '']

                col1, col2 = st.columns([1.5, 1])

                with col1:
                    model_names = metrics_df['ModelName'].tolist()
                    fig_bar = go.Figure()
                    for metric, color, label in [
                        ('Accuracy', '#4ECDC4', '准确率'),
                        ('Macro_F1', '#FFE66D', '宏平均F1'),
                        ('Macro_AUC', '#FF6B6B', '宏平均AUC')
                    ]:
                        if metric not in metrics_df.columns:
                            continue
                        vals = pd.to_numeric(metrics_df[metric], errors='coerce').fillna(0)
                        fig_bar.add_trace(go.Bar(
                            name=label,
                            x=model_names,
                            y=vals,
                            marker_color=color,
                            text=vals.round(3),
                            textposition='auto',
                        ))
                    fig_bar.update_layout(
                        barmode='group',
                        title=dict(text='四模型性能对比', font=dict(color='#ADB5BD', size=14)),
                        xaxis=dict(title='模型', tickfont=dict(color='#ADB5BD'), gridcolor='#2D2D3D'),
                        yaxis=dict(title='得分', range=[0, 1], tickfont=dict(color='#ADB5BD'),
                                   gridcolor='#2D2D3D'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='white')),
                        height=400,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col2:
                    try:
                        rf_imp, gbdt_imp = load_feature_importance()
                        imp = rf_imp if rf_imp is not None else gbdt_imp
                        if imp is not None and len(imp) > 0:
                            imp_top = imp.head(10)
                            x_col = [c for c in imp_top.columns if c != imp_top.columns[0]][0]
                            y_col = imp_top.columns[0]
                            fig_imp = go.Figure(go.Bar(
                                x=pd.to_numeric(imp_top[x_col], errors='coerce').fillna(0),
                                y=imp_top[y_col],
                                orientation='h',
                                marker_color='#00B4D8',
                                text=imp_top[y_col],
                                textposition='outside',
                            ))
                            fig_imp.update_layout(
                                title=dict(text='随机森林特征重要性 (Top 10)', font=dict(color='#ADB5BD', size=14)),
                                xaxis=dict(title='重要性', tickfont=dict(color='#ADB5BD'), gridcolor='#2D2D3D'),
                                yaxis=dict(tickfont=dict(color='#ADB5BD'), gridcolor='#2D2D3D'),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=400,
                                margin=dict(l=150, r=40, t=40, b=30),
                            )
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.info("特征重要性数据暂不可用")
                    except Exception:
                        st.info("特征重要性图表暂不可用")
            else:
                st.info("模型对比数据未找到，请确保数据文件存在且包含 ModelName 列。")
                if metrics_df is not None:
                    with st.expander("查看原始数据表格"):
                        st.dataframe(metrics_df)
        except Exception as e:
            st.error(f"模型对比加载失败: {str(e)}")