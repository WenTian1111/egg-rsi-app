# Egg Roll Stability Analysis System 🥚

A machine learning-based intelligent prediction system for egg roll stability (RSI). Upload egg photos to automatically extract 19 morphological features and predict risk levels in real-time using SVM/RF/GBDT models.

## Live Demo (No Installation Required)

👉 **https://egg-rsi-app.streamlit.app**

---

## Quick Start (From Source Package)

### Requirements
- Python 3.9 or higher
- No internet connection needed (fully offline)

### Steps

```bash
# 1. Extract the package
tar -xzf egg-repo.tar.gz
cd egg-repo

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Quick Start (From GitHub)

```bash
git clone https://github.com/WenTian1111/egg-rsi-app.git
cd egg-rsi-app
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## How to Use

### Tab 1: Showcase (展板模式)
- Browse all 90 eggs from the database
- View the **image processing pipeline** (original → grayscale → mask → contour)
- Explore **feature analysis** with radar charts
- Compare **model performance** (SVM / RF / GBDT / LR)
- View **feature importance** (RF vs GBDT dual-bar comparison)

### Tab 2: Prediction (预测模式)

**Option A — Upload a new photo:**
1. Take a photo of an egg (top-down view on a blue background)
2. Upload the image
3. The system automatically: HSV segmentation → 19D feature extraction → model prediction
4. View: risk level 🔴🟡🟢 + probability distribution

**Option B — Quick select from database:**
1. Pick an egg from the 90-sample database
2. Instantly see its prediction result

### Model Switching
- Use the sidebar to switch between **SVM**, **RF**, **GBDT**, and **LR**
- SVM is the default optimal model (macro-average AUC: 85.85%)

---

## Package Contents

| Path | Description |
|:-----|:------------|
| `app.py` | Main application entry point |
| `pages/showcase.py` | Showcase page (database, features, model comparison) |
| `pages/prediction.py` | Prediction page (upload & quick select) |
| `utils/data_loader.py` | Data loading & column mapping |
| `utils/feature_extraction.py` | OpenCV feature extraction (HSV pipeline) |
| `model/svm_model.joblib` | Pretrained SVM model |
| `model/rf_model.joblib` | Pretrained Random Forest model |
| `model/gbdt_model.joblib` | Pretrained GBDT model |
| `model/lr_model.joblib` | Pretrained Logistic Regression model |
| `model/scaler.joblib` | Feature scaler |
| `data/fusion_dataset.csv` | 270-sample fused static-dynamic dataset |
| `data/egg_images/` | Contour images of all 90 eggs |
| `requirements.txt` | Python dependencies |

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **Vision**: OpenCV (HSV segmentation + feature extraction)
- **Models**: scikit-learn (SVM / RF / GBDT / LR)
- **UI**: Dark tech-blue themed CSS animations

## Project Info

- **Project**: Southwest University Innovation & Entrepreneurship Training Program (S202510635378)
- **Researcher**: Yang Han
- **Advisor**: Li Changying
