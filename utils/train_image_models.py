"""
Train image-based ML models for egg RSI prediction.
Extracts features from contour JPGs (same pipeline as uploaded photos).
Trains 4 models: LR, SVM, RF, GBDT.
"""
import cv2, numpy as np, pandas as pd, os, sys, json

# Add project root to path
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils.feature_extraction import _compute_features
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


def extract_features_from_contour(egg_id):
    """Extract 19 features from contour JPG (same as uploaded photo pipeline)."""
    img_path = os.path.join(DATA_DIR, 'egg_images', f'{egg_id}号鸡蛋轮廓.jpg')
    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    corners = [binary[5, 5], binary[5, w-5], binary[h-5, 5], binary[h-5, w-5]]
    if sum(p > 127 for p in corners) >= 2:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    return _compute_features(mask, largest)


def main():
    print("=" * 60)
    print("  图像版鸡蛋RSI预测模型训练")
    print("=" * 60)

    # Step 1: Load CSV labels
    df = pd.read_csv(CSV_PATH)
    print(f"\n[1] 加载数据: {df.shape}")
    print(f"    鸡蛋数: {df['EggID'].nunique()}")
    print(f"    试验数: {len(df)}")

    # Step 2: Extract features from contour images for each trial
    print(f"\n[2] 从轮廓图提取特征（共 {len(df)} 条试验）...")
    X_list, y_list = [], []
    egg_count = len(df['EggID'].unique())
    processed = 0

    for _, row in df.iterrows():
        egg_id = int(row['EggID'])
        features = extract_features_from_contour(egg_id)
        if features is not None:
            X_list.append([features[f] for f in STATIC_FEATURES_19])
            y_list.append(int(row['RSI_GroupNum']))
            processed += 1

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"    提取成功: {processed}/{len(df)} 条试验")
    print(f"    X: {X.shape}, y: {y.shape}")
    print(f"    类别分布: {np.bincount(y)}")

    # Step 3: Train-test split
    print(f"\n[3] 划分训练/测试集 (80/20, 分层)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    训练集: {X_train.shape[0]} 样本")
    print(f"    测试集: {X_test.shape[0]} 样本")
    print(f"    训练集分布: {np.bincount(y_train)}")
    print(f"    测试集分布: {np.bincount(y_test)}")

    # Step 4: Scale features
    print(f"\n[4] Z-score 标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train models
    print(f"\n[5] 训练模型...")
    models = {
        'LR': LogisticRegression(
            C=1.0, max_iter=1000,
            random_state=42, class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale',
            probability=True, random_state=42, class_weight='balanced'
        ),
        'RF': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'GBDT': GradientBoostingClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            learning_rate=0.1, random_state=42
        ),
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

        results[name] = {
            'accuracy': round(acc, 4),
            'macro_f1': round(f1, 4),
            'macro_auc': round(auc, 4),
        }
        print(f"    {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # Save model
        path = os.path.join(MODEL_DIR, f'{name.lower()}_model.joblib')
        joblib.dump(model, path)
        print(f"      已保存: {path}")

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\n[6] 已保存 Scaler: {scaler_path}")

    # Save results
    with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    已保存结果: model_results.json")

    best = max(results, key=lambda k: results[k]['macro_f1'])
    print(f"\n{'=' * 60}")
    print(f"  最优模型: {best} (F1={results[best]['macro_f1']})")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
