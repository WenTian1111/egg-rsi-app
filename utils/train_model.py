"""Train ML models on the egg fusion dataset for RSI prediction."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'data', 'fusion_dataset.csv')
MODEL_DIR = os.path.join(BASE, 'model')

STATIC_FEATURES = [
    'Static_Area_像素面积', 'Static_Perimeter_轮廓周长',
    'Static_MajorAxisLength_长轴像素长度', 'Static_MinorAxisLength_短轴像素长度',
    'Static_EquivalentDiameter_等效圆直径', 'Static_Eccentricity_离心率',
    'Static_ShapeIndex_机器视觉ESI',
    'Static_Circularity_圆形度', 'Static_Solidity_坚实度',
    'Static_Extent_延展度',
    'Static_AsymmetryIndex_不对称指数', 'Static_MajorAxisOffsetRatio_长轴偏移率',
    'Static_Hu1', 'Static_Hu2', 'Static_Hu3', 'Static_Hu4',
    'Static_Hu5', 'Static_Hu6', 'Static_Hu7'
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[STATIC_FEATURES].copy()
    y = df['RSI_GroupNum'].copy()
    return X, y, df


def train_and_save():
    X, y, df = load_data()
    print(f"Data loaded: {X.shape}, classes: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'RF': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'GBDT': GradientBoostingClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            learning_rate=0.1, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale',
            probability=True, random_state=42, class_weight='balanced'
        ),
    }

    results = {}
    for name, model in models.items():
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = 0.0

        results[name] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'macro_auc': round(auc, 4),
        }
        print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        model_path = os.path.join(MODEL_DIR, f'{name.lower()}_model.joblib')
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")

    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")

    with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda k: results[k]['macro_f1'])
    print(f"\nBest model: {best} (F1={results[best]['macro_f1']})")
    return results


if __name__ == '__main__':
    train_and_save()
