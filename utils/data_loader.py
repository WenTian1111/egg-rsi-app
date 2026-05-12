"""Data loading utilities for the Egg RSI Streamlit app."""
import pandas as pd
import json
import joblib
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'model')

# EXACT 19 features from MATLAB p4.m
STATIC_FEATURES_19 = [
    'Static_Area_像素面积', 'Static_Perimeter_轮廓周长',
    'Static_MajorAxisLength_长轴像素长度', 'Static_MinorAxisLength_短轴像素长度',
    'Static_EquivalentDiameter_等效圆直径', 'Static_Eccentricity_离心率',
    'Static_ShapeIndex_机器视觉ESI',
    'Static_Circularity_圆形度', 'Static_Solidity_坚实度', 'Static_Extent_延展度',
    'Static_AsymmetryIndex_不对称指数', 'Static_MajorAxisOffsetRatio_长轴偏移率',
    'Static_Hu1', 'Static_Hu2', 'Static_Hu3', 'Static_Hu4',
    'Static_Hu5', 'Static_Hu6', 'Static_Hu7'
]

FEATURE_LABELS = {
    'Static_ShapeIndex_机器视觉ESI': '蛋形指数 (ESI)',
    'Static_AsymmetryIndex_不对称指数': '不对称指数',
    'Static_Eccentricity_离心率': '离心率',
    'Static_Area_像素面积': '像素面积',
    'Static_Perimeter_轮廓周长': '轮廓周长',
    'Static_Circularity_圆形度': '圆形度',
    'Static_Solidity_坚实度': '坚实度',
    'Static_Extent_延展度': '延展度',
    'Static_MajorAxisLength_长轴像素长度': '长轴长度',
    'Static_MinorAxisLength_短轴像素长度': '短轴长度',
    'Static_EquivalentDiameter_等效圆直径': '等效直径',
    'Static_MajorAxisOffsetRatio_长轴偏移率': '长轴偏移率',
}

RSI_LABELS = {1: ('低风险', '#4ECDC4', '🟢'), 2: ('中风险', '#FFE66D', '🟡'), 3: ('高风险', '#FF6B6B', '🔴')}
MODEL_NAMES = {'svm': 'SVM (支持向量机)', 'rf': 'RF (随机森林)', 'gbdt': 'GBDT (梯度提升)', 'lr': 'LR (逻辑回归)'}

# Model metrics from MATLAB p4.m for display
PAPER_METRICS = {
    'SVM': {'Accuracy': 0.741, 'Macro_F1': 0.816, 'Macro_AUC': 0.858},
    'RF': {'Accuracy': 0.722, 'Macro_F1': 0.804, 'Macro_AUC': 0.862},
    'GBDT': {'Accuracy': 0.722, 'Macro_F1': 0.804, 'Macro_AUC': 0.866},
    'LR': {'Accuracy': 0.593, 'Macro_F1': 0.666, 'Macro_AUC': 0.797},
}

FEATURE_PIPELINE_STEPS = [
    ('原图', 'original'),
    ('灰度图', 'grayscale'),
    ('二值掩膜', 'mask'),
    ('轮廓质心包围盒', 'contour'),
]


def load_fusion_data():
    """Load fusion dataset with normalized column names."""
    xlsx_path = os.path.join(DATA_DIR, 'Egg_Fusion_Dataset.xlsx')
    csv_path = os.path.join(DATA_DIR, 'fusion_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        return None

    # Normalize column names: EggID → egg_id, RSI_GroupNum → RSI
    rename_map = {}
    if 'EggID' in df.columns:
        rename_map['EggID'] = 'egg_id'
    if 'RSI_GroupNum' in df.columns:
        rename_map['RSI_GroupNum'] = 'RSI'
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_model_metrics():
    """Return paper metrics as DataFrame."""
    rows = []
    for model, metrics in PAPER_METRICS.items():
        rows.append({'ModelName': model, **metrics})
    return pd.DataFrame(rows)


def load_feature_importance():
    """Load feature importance from consolidated CSV (RF + GBDT)."""
    path = os.path.join(DATA_DIR, 'feature_importance_RF_GBDT.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    # Fallback to old separate files
    rf_path = os.path.join(DATA_DIR, 'RF_feature_importance.csv')
    gbdt_path = os.path.join(DATA_DIR, 'GBDT_feature_importance.csv')
    rf = pd.read_csv(rf_path) if os.path.exists(rf_path) else None
    gbdt = pd.read_csv(gbdt_path) if os.path.exists(gbdt_path) else None
    return rf, gbdt


def load_model(model_name='svm'):
    path = os.path.join(MODEL_DIR, f'{model_name.lower()}_model.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_scaler():
    path = os.path.join(MODEL_DIR, 'scaler.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_model_results():
    path = os.path.join(MODEL_DIR, 'model_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_egg_image_path(egg_id):
    return os.path.join(DATA_DIR, 'egg_images', f'{egg_id}号鸡蛋轮廓.jpg')


def generate_pipeline_images(egg_id):
    """
    Generate 4-step pipeline images from the egg contour image.
    Returns dict of step_key → numpy array (BGR).
    """
    img_path = get_egg_image_path(egg_id)
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # Original
    original = img.copy()

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Binary mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure white foreground on black bg
    corner_pixels = [binary[5, 5], binary[5, w-5], binary[h-5, 5], binary[h-5, w-5]]
    if sum(p > 127 for p in corner_pixels) >= 2:
        binary = cv2.bitwise_not(binary)
    mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Contour + centroid + bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_viz = original.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_viz, [largest], -1, (255, 0, 0), 3)
        M = cv2.moments(binary)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(contour_viz, (cx, cy), 8, (255, 0, 0), -1)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(contour_viz, (x, y), (x + bw, y + bh), (255, 255, 0), 3)

    return {
        'original': original,
        'grayscale': gray_bgr,
        'mask': mask_bgr,
        'contour': contour_viz,
    }


def predict_risk(features_dict, model_name='svm'):
    """Predict RSI risk using the 19-feature model. Returns (prediction, probabilities)."""
    model = load_model(model_name)
    scaler = load_scaler()
    if model is None or scaler is None:
        return None, None

    df = pd.DataFrame([features_dict])
    for col in STATIC_FEATURES_19:
        if col not in df.columns:
            df[col] = 0.0
    df = df[STATIC_FEATURES_19]

    if model_name in ['svm', 'lr']:
        features_scaled = scaler.transform(df.values)
        y_pred = model.predict(features_scaled)
        y_prob = model.predict_proba(features_scaled)
    else:
        y_pred = model.predict(df.values)
        y_prob = model.predict_proba(df.values)

    return int(y_pred[0]), y_prob[0].tolist()
