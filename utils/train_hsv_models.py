"""
Train ML models using the HSV segmentation pipeline (same as uploaded photos).
This ensures training and inference features are from the same distribution.
"""
import cv2, numpy as np, pandas as pd, os, sys, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.data_loader import STATIC_FEATURES_19
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'model')
CSV_PATH = os.path.join(DATA_DIR, 'fusion_dataset.csv')


def extract_hsv_features(egg_id):
    """Extract features using process_uploaded_image pipeline (HSV segmentation)."""
    img_path = os.path.join(DATA_DIR, 'egg_images', f'{egg_id}号鸡蛋轮廓.jpg')
    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None

    # === Exact same HSV pipeline as process_uploaded_image() ===
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0

    s_uint8 = (s_channel * 255).astype(np.uint8)
    tS = cv2.threshold(s_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    s_thresh = max(0.10, min(0.60, tS * 1.05))
    mask_s = s_channel < s_thresh

    v_uint8 = (v_channel * 255).astype(np.uint8)
    tV = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    v_thresh = min(0.95, max(0.30, tV * 0.85))
    mask_v = v_channel > v_thresh

    binary_mask = np.logical_and(mask_s, mask_v).astype(np.uint8) * 255

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # === Feature computation (same _compute_features) ===
    from utils.feature_extraction import _compute_features
    return _compute_features(clean_mask, largest_contour)


def main():
    print("=" * 60)
    print("  HSV流水线版 鸡蛋RSI预测模型训练")
    print("  训练和推理使用完全相同的特征提取")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\n[1] 加载数据: {df.shape}, 鸡蛋={df['EggID'].nunique()}")

    print(f"\n[2] HSV流水线提取特征（共{len(df)}条试验）...")
    X_list, y_list = [], []
    success = 0
    for _, row in df.iterrows():
        egg_id = int(row['EggID'])
        features = extract_hsv_features(egg_id)
        if features is not None:
            X_list.append([features[f] for f in STATIC_FEATURES_19])
            y_list.append(int(row['RSI_GroupNum']))
            success += 1

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"    成功: {success}/{len(df)}")
    print(f"    X: {X.shape}, y: {y.shape}")
    print(f"    类别分布: {np.bincount(y)}")

    print(f"\n[3] 训练/测试划分 (80/20, 分层)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    训练: {X_train.shape[0]}, 测试: {X_test.shape[0]}")

    print(f"\n[4] 标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[5] 训练模型...")
    models = {
        'LR': LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1),
        'GBDT': GradientBoostingClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name in ['LR', 'SVM']:
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

        results[name] = {'accuracy': round(acc, 4), 'macro_f1': round(f1, 4), 'macro_auc': round(auc, 4)}
        print(f"    {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        path = os.path.join(MODEL_DIR, f'{name.lower()}_model.joblib')
        joblib.dump(model, path)
        print(f"      已保存: {path}")

    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\n[6] Scaler已保存: {scaler_path}")

    with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda k: results[k]['macro_f1'])
    print(f"\n{'=' * 60}")
    print(f"  最优模型: {best} (F1={results[best]['macro_f1']})")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
