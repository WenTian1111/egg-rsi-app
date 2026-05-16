# Egg Roll Stability Analysis System 🥚

A machine learning-based intelligent prediction system for egg roll stability (RSI). Upload egg photos to automatically extract 19 morphological features and predict risk levels in real-time using SVM/RF/GBDT models.

## Live Demo

👉 **https://egg-rsi-app.streamlit.app**

---

## Quick Start (From Source Package)

### Requirements
- Python 3.9 or higher
- No internet required (fully offline, model included in package)

### Steps

```bash
# 1. Open terminal in the Egg_RSI_App folder

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

### Prerequisites
- [Git LFS](https://git-lfs.com/) must be installed (required for the 168MB U2Net model)

```bash
# Install Git LFS first, then:
git clone https://github.com/WenTian1111/egg-rsi-app.git
cd egg-rsi-app
git lfs pull
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## How to Use

### Tab 1: Showcase (Database & Analysis)
- Browse all 90 eggs from the database
- View the **image processing pipeline** (original → grayscale → mask → contour)
- Explore **feature analysis** with radar charts
- Compare **model performance** (SVM / RF / GBDT / LR)
- View **feature importance** (RF vs GBDT dual-bar comparison)
- Expandable thesis figures (experiment platform, risk validation)

### Tab 2: Prediction (Upload & Classify)

**Option A — Upload a new photo:**
1. Take a photo of an egg (top-down view on a contrasting background)
2. Upload the image
3. The system automatically segments → extracts 19 features → predicts risk level
4. View: risk level 🔴🟡🟢 + probability distribution + strategy used

**Option B — Quick select from database:**
1. Pick an egg from the 90-sample database
2. Instantly see its prediction result

### Model Switching
- Switch between **SVM**, **RF**, **GBDT**, and **LR** in the sidebar
- SVM is the default optimal model (macro-average AUC: 85.85%)

---

## Package Contents

| Path | Description |
|:-----|:------------|
| `app.py` | Main application entry point + CSS theme |
| `pages/showcase.py` | Showcase page (pipeline, features, model comparison, paper figures) |
| `pages/prediction.py` | Prediction page (upload & quick select + thesis images) |
| `utils/data_loader.py` | Data loading & column mapping |
| `utils/feature_extraction.py` | U2Net (OpenCV DNN) + 7-strategy multi-algorithm segmentation |
| `model/u2net.onnx` | Pre-trained U2Net model (168MB, via Git LFS) |
| `model/svm_model.joblib` | Pre-trained SVM model |
| `model/rf_model.joblib` | Pre-trained Random Forest model |
| `model/gbdt_model.joblib` | Pre-trained GBDT model |
| `model/lr_model.joblib` | Pre-trained Logistic Regression model |
| `model/scaler.joblib` | Feature scaler |
| `data/fusion_dataset.csv` | 270-sample fused static-dynamic dataset |
| `data/egg_images/` | Contour images of all 90 eggs |
| `data/feature_importance_RF_GBDT.csv` | RF + GBDT feature importance table |
| `assets/` | Thesis figures (experiment platform, RSI 3D cluster, trajectory tracking, etc.) |
| `requirements.txt` | Python dependencies |

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **Vision**: U2Net (OpenCV DNN, 168MB) + OpenCV (7-strategy multi-algorithm competition)
- **Models**: scikit-learn (SVM / RF / GBDT / LR)
- **UI**: Sci-fi dark tech theme — hexagonal grid, Tron corners, circuit traces, data stream animations

## Project Info

- **Project**: Southwest University Innovation & Entrepreneurship Training Program (S202510635378)
- **Researcher**: Yang Han
- **Advisor**: Li Changying
