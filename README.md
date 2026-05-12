# 鸡蛋滚落稳定性分析系统 🥚

基于机器学习的鸡蛋滚落稳定性（RSI）智能预测系统。上传鸡蛋照片，自动提取 19 维形态特征，通过 SVM/RF/GBDT 模型实时预测风险等级。

## 功能

- **📸 上传新图片** — 上传鸡蛋照片 → HSV 分割 → 特征提取 → 模型预测
- **🎯 快速选择** — 从 90 个鸡蛋数据库中快速选择样本
- **🔬 数据库展示** — 图像处理流水线可视化、特征雷达图、模型性能对比
- **📊 特征重要性** — RF vs GBDT 特征重要性双柱对比

## 技术栈

- **前端**: Streamlit + Plotly
- **视觉**: OpenCV (HSV 分割 + 特征提取)
- **模型**: scikit-learn (SVM / RF / GBDT / LR)
- **UI**: 深蓝科技风 CSS 动画

## 本地运行

```bash
git clone https://github.com/WenTian1111/egg-rsi-app.git
cd egg-rsi-app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 在线演示

[Streamlit Cloud](https://egg-rsi-app.streamlit.app)
