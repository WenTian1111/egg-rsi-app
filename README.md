# Egg Roll Stability Analysis System 🥚

A machine learning-based intelligent prediction system for egg roll stability (RSI). Upload egg photos to automatically extract 19 morphological features and predict risk levels in real-time using SVM/RF/GBDT models.

## Features

- **📸 Upload New Image** — Upload egg photo → HSV segmentation → feature extraction → model prediction
- **🎯 Quick Select** — Rapidly select samples from a database of 90 eggs
- **🔬 Database View** — Image processing pipeline visualization, feature radar charts, model performance comparison
- **📊 Feature Importance** — Dual-bar comparison of RF vs GBDT feature importance

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **Vision**: OpenCV (HSV segmentation + feature extraction)
- **Models**: scikit-learn (SVM / RF / GBDT / LR)
- **UI**: Dark tech-blue themed CSS animations

## Running Locally

```bash
git clone https://github.com/WenTian1111/egg-rsi-app.git
cd egg-rsi-app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

[Streamlit Cloud](https://egg-rsi-app.streamlit.app)
