"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic.

Segmentation strategy (multi-layered fallback):
  PRIMARY:    GrabCut 固定中心矩形 (原V3方案, 居中鸡蛋最优)
  Fallback 1: 灰度 Otsu + 连通筛选 (检测鸡蛋亮/暗方向)
  Fallback 2: 渐进腐蚀法 (类二值图)
  Fallback 3: L*a*b* + b通道 (彩色/复杂背景)
  Fallback 4: 边缘引导 GrabCut (自适应矩形)
  Fallback 5: Canny 边缘检测 (最后手段)
"""
import cv2
import numpy as np
import os

# U2Net 深度学习分割（OpenCV DNN 本地推理，零额外依赖）
_u2net_net = None
_u2net_available = None  # None=未检测, True/False

def _is_u2net_ready():
    """检测 U2Net ONNX 模型是否可用（纯本地，无需联网/下载）。"""
    global _u2net_available, _u2net_net
    if _u2net_available is not None:
        return _u2net_available
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'model', 'u2net.onnx')
    if not os.path.exists(model_path):
        _u2net_available = False
        return False
    try:
        _u2net_net = cv2.dnn.readNetFromONNX(model_path)
        _u2net_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _u2net_available = True
    except Exception:
        _u2net_available = False
    return _u2net_available


def _segment_by_u2net(img_in):
    """Strategy J: U2Net 深度学习分割（OpenCV DNN 本地推理）。
    
    使用预训练的 U²-Net 模型（u2net, 168MB）做像素级精确分割。
    完全本地运行，无需联网下载，零额外依赖。
    对阴影、光照变化、复杂背景都鲁棒。
    """
    if not _is_u2net_ready():
        return None, None, False

    h, w = img_in.shape[:2]

    # 预处理：缩放到 320x320 + 归一化
    blob = cv2.dnn.blobFromImage(img_in, 1/255.0, (320, 320),
                                 (0.485, 0.456, 0.406), swapRB=False,
                                 crop=False)
    # 用训练集的 std 做归一化
    blob[:, 0, :, :] = (blob[:, 0, :, :] - 0.485) / 0.229
    blob[:, 1, :, :] = (blob[:, 1, :, :] - 0.456) / 0.224
    blob[:, 2, :, :] = (blob[:, 2, :, :] - 0.406) / 0.225

    # 推理
    _u2net_net.setInput(blob)
    output = _u2net_net.forward()
    # 如果输出是 4D (N,C,H,W)，降维到 2D
    if output.ndim == 4:
        prob = output[0, 0, :, :]
    else:
        prob = output[0]

    # 后处理
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    _, mask_bin = cv2.threshold(prob, 0.5, 255, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype(np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask_bin)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    clean = _fill_holes(clean)
    cnt_out, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not cnt_out:
        return None, None, False
    return clean, max(cnt_out, key=cv2.contourArea), True


# ═══════════════════════════════════════════════
# FALLBACK 2: 渐进腐蚀法
# ═══════════════════════════════════════════════

def _segment_by_binary_erosion(img):
    """Strategy C: 针对类二值图的渐进腐蚀法.
    逐步腐蚀直到鸡蛋与边界断开 → 取最大组件 → 膨胀恢复.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    for iters in range(3, 50, 2):
        eroded = cv2.erode(binary, kernel, iterations=iters)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        best_label, best_area = None, 0
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            if area / (h * w) < 0.01:
                continue
            touches_border = (x <= 0 or y <= 0 or x + cw >= w or y + ch >= h)
            if touches_border:
                continue
            if area > best_area:
                best_area = area
                best_label = i
        if best_label is not None:
            clean_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
            clean_mask = cv2.dilate(clean_mask, kernel, iterations=iters)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            clean_mask = _fill_holes(clean_mask)
            cnt2, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnt2:
                return clean_mask, max(cnt2, key=cv2.contourArea), True
    return None, None, False


# ═══════════════════════════════════════════════
# FALLBACK 3: L*a*b* 颜色分割
# ═══════════════════════════════════════════════

def _segment_by_lab(img):
    """Strategy D: L*a*b* 色彩空间分割 (彩色/复杂背景)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    lab_eq_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    lab_eq_lab2 = cv2.cvtColor(lab_eq_rgb, cv2.COLOR_BGR2LAB)
    _, _, b_eq = cv2.split(lab_eq_lab2)

    b_blur = cv2.GaussianBlur(b_eq, (7, 7), 0)
    _, b_mask = cv2.threshold(b_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center_fg = np.mean(b_mask[h//3:2*h//3, w//3:2*w//3] > 0)
    if center_fg < 0.3:
        b_mask = 255 - b_mask
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    result = _clean_connected_components(b_mask, h, w)
    if result[0] is not None:
        m, _, _ = result
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        m2 = _fill_holes(m2)
        cnt2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt2:
            return m2, max(cnt2, key=cv2.contourArea), True
    return None, None, False


# ═══════════════════════════════════════════════
# FALLBACK 4: 边缘引导 GrabCut
# ═══════════════════════════════════════════════

def _segment_by_edge_guided_grabcut(img):
    """Strategy E: 边缘引导 GrabCut (修复崩溃版)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    margin_x = max(30, int(cw * 0.30))
    margin_y = max(30, int(ch * 0.30))
    rx = max(0, x - margin_x)
    ry = max(0, y - margin_y)
    rw = min(w - rx, cw + 2 * margin_x)
    rh = min(h - ry, ch + 2 * margin_y)
    if rw < 40 or rh < 40:
        return None, None, False
    roi = gray[ry:ry+rh, rx:rx+rw]
    if roi.size == 0 or np.std(roi) < 5:
        return None, None, False
    rect = (rx, ry, rw, rh)

    try:
        mask_gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, None, False

    bin_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.uint8) * 255
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bin_mask2, largest = _keep_largest_contour(bin_mask)
    if bin_mask2 is None:
        return None, None, False
    cnt2, _ = cv2.findContours(bin_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt2:
        return None, None, False
    return bin_mask2, max(cnt2, key=cv2.contourArea), True


# ═══════════════════════════════════════════════
# FALLBACK 5: Canny 边缘检测
# ═══════════════════════════════════════════════

def _segment_by_edge(img):
    """Strategy F: Canny 边缘检测 (最后手段)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    binary, largest = _keep_largest_contour(dilated)
    if binary is None:
        return None, None, False
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return binary, largest, True


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def _fill_holes(mask):
    """用 floodfill 填充二值掩膜中的小孔。"""
    h, w = mask.shape
    mask_ext = np.zeros((h + 2, w + 2), np.uint8)
    mask_ext[1:-1, 1:-1] = mask
    cv2.floodFill(mask_ext, None, (0, 0), 255)
    mask_inv = 255 - mask_ext[1:-1, 1:-1]
    return mask | mask_inv


def _keep_largest_contour(binary):
    """保留二值图中最大的连通域。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean, largest


def _clean_connected_components(binary, h, w):
    """清理二值图: 去掉小区域 + 去除连接到边界的组件。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_label, best_area = None, 0
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area / (h * w) < 0.02:
            continue
        touches_border = (x <= 0 or y <= 0 or x + cw >= w or y + ch >= h)
        if touches_border:
            continue
        if area > best_area:
            best_area = area
            best_label = i
    if best_label is None:
        return None, None, False
    binary = np.where(labels == best_label, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    binary = _fill_holes(binary)
    cnt2, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt2:
        return None, None, False
    return binary, max(cnt2, key=cv2.contourArea), True


def _evaluate_mask_quality(mask):
    """评估分割质量。返回 0-1 分数。"""
    area = cv2.countNonZero(mask)
    h, w = mask.shape
    total = h * w
    ratio = area / total if total > 0 else 0
    if ratio < 0.01 or ratio > 0.90:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return 0.0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = min(1.0, area / max(hull_area, 1))
    if solidity < 0.2:
        return 0.0
    return min(1.0, solidity)


def _segment_by_kmeans_photo(img_in):
    """Strategy H: 自适应 K-means 聚类分割（照片专用）。
    
    核心思想：先用 k=3 聚类 + 智能合并，处理光照不均匀导致一侧缺失的问题。
    如果 k=3 质量不佳（对简单背景可能过分割），自动回退到 k=2。
    
    流程：
    1. 双边滤波（去噪保边）
    2. k=3 聚类 → 按中心覆盖率排序 → 合并覆盖率>20%的集群
    3. 评估质量，如果好 → 返回
    4. 回退到 k=2 聚类 → 返回
    """
    h, w = img_in.shape[:2]
    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        small = cv2.resize(img_in, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_in
    sh, sw = small.shape[:2]

    # 双边滤波
    bilateral = cv2.bilateralFilter(small, 7, 50, 50)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    cx, cy = sw // 2, sh // 2
    margin = min(sw, sh) // 4
    x1, y1 = max(0, cx - margin), max(0, cy - margin)
    x2, y2 = min(sw, cx + margin), min(sh, cy + margin)

    def _kmeans_and_mask(n_clusters, merge_thresh=0.20):
        """内部函数：做 K-means 聚类并生成掩膜"""
        _, labels, _ = cv2.kmeans(pixels, n_clusters, None, criteria, 10,
                                  cv2.KMEANS_PP_CENTERS)
        label_img = labels.reshape(sh, sw)

        # 计算每个 cluster 的中心覆盖率
        center_region = label_img[y1:y2, x1:x2].flatten()
        counts = np.bincount(center_region, minlength=n_clusters)
        ratios = counts / max(counts.sum(), 1)

        # 按覆盖率排序
        ranked = sorted([(ratios[i], i) for i in range(n_clusters)], reverse=True)

        # 智能合并：主集群 + 覆盖率>阈值的所有集群
        egg_clusters = [ranked[0][1]]
        for i in range(1, n_clusters):
            if ranked[i][0] > merge_thresh:
                egg_clusters.append(ranked[i][1])

        mask = np.zeros((sh, sw), dtype=np.uint8)
        for cl in egg_clusters:
            mask[label_img == cl] = 255

        # 极小核形态学
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8), iterations=1)
        return mask

    # 先试 k=3 + 合并
    mask = _kmeans_and_mask(3, merge_thresh=0.20)

    # 评估质量
    small_quality = _evaluate_mask_quality(mask)
    if small_quality < 0.95:  # k=3 质量不好 → 回退到 k=2
        mask = _kmeans_and_mask(2, merge_thresh=1.0)  # merge_thresh=1.0 只选第一个

    if scale < 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    clean = _fill_holes(clean)
    cnt_out, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not cnt_out:
        return None, None, False
    return clean, max(cnt_out, key=cv2.contourArea), True


def _segment_by_edge_floodfill(img_in):
    """Strategy I: 边缘检测 + 膨胀闭合 + 洪水填充。
    基于硬边缘（Canny），不受阴影影响。
    适用于鸡蛋和背景颜色相似但边缘清晰的场景。
    """
    h, w = img_in.shape[:2]
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny 边缘检测（低阈值让边缘更完整）
    edges = cv2.Canny(blurred, 20, 80)

    # 多轮膨胀闭合断开的边缘
    closed = edges.copy()
    for _ in range(3):
        closed = cv2.dilate(closed, np.ones((5, 5), np.uint8))
    closed = cv2.erode(closed, np.ones((3, 3), np.uint8), iterations=3)

    # 从中心洪水填充
    flooded = closed.copy()
    mask_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, mask_fill, (w // 2, h // 2), 255)

    # 保留最大连通域
    contours, _ = cv2.findContours(flooded, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(flooded)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    mask = _fill_holes(mask)

    cnt_out, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not cnt_out:
        return None, None, False
    return mask, max(cnt_out, key=cv2.contourArea), True


# ═══════════════════════════════════════════════
# PRIMARY: GrabCut 固定中心矩形
# ═══════════════════════════════════════════════

def _segment_by_grabcut_center(img):
    """Strategy A (PRIMARY): 固定中心矩形 GrabCut.
    先降采样到 max(w,h) <= 768 加速, 再恢复.
    """
    h, w = img.shape[:2]
    scale = 768.0 / max(h, w) if max(h, w) > 768 else 1.0
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        sh, sw = small.shape[:2]
    else:
        small = img
        sh, sw = h, w

    margin_x = max(20, int(sw * 0.10))
    margin_y = max(20, int(sh * 0.10))
    rect = (margin_x, margin_y, sw - 2*margin_x, sh - 2*margin_y)
    if rect[2] < 40 or rect[3] < 40:
        return None, None, False

    try:
        mask_gc = np.zeros((sh, sw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, None, False

    bin_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.uint8) * 255
    if scale < 1.0:
        bin_mask = cv2.resize(bin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bin_mask = _fill_holes(bin_mask)
    binary, largest = _keep_largest_contour(bin_mask)
    if binary is None:
        return None, None, False
    return binary, largest, True


# ═══════════════════════════════════════════════
# FALLBACK 1: 灰度 Otsu + 连通筛选
# ═══════════════════════════════════════════════

def _segment_by_grayscale_otsu(img):
    """Strategy B: 灰度 Otsu 阈值分割.
    自动检测鸡蛋是亮是暗，同时试 BINARY 和 BINARY_INV.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    best_mask, best_contour = None, None
    for use_inv in [True, False]:
        flag = cv2.THRESH_BINARY_INV if use_inv else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask, cont, ok = _clean_connected_components(binary, h, w)
        if ok:
            q = _evaluate_mask_quality(mask)
            if q > 0.1:
                if best_mask is None or q > _evaluate_mask_quality(best_mask):
                    best_mask, best_contour = mask, cont

    if best_mask is not None:
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        best_mask = _fill_holes(best_mask)
        cnt2, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnt2:
            return best_mask, max(cnt2, key=cv2.contourArea), True
    return None, None, False


# ═══════════════════════════════════════════════
# HELPERS

def _segment_by_contour_floodfill(img):
    """Strategy G: 阈值提取轮廓 → 膨胀闭合 → 洪水填充。
    专为白线黑底的鸡蛋轮廓图设计。先提取明亮轮廓线，
    通过膨胀连接断点，再从中心洪水填充得到完整蛋形。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((11, 11), np.uint8), iterations=5)
    flooded = dilated.copy()
    mask_fill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flooded, mask_fill, (w // 2, h // 2), 255)
    clean = cv2.morphologyEx(flooded, cv2.MORPH_OPEN,
                             np.ones((15, 15), np.uint8))
    clean = _fill_holes(clean)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(clean)
    cv2.drawContours(result, [largest], -1, 255, -1)
    return result, largest, True


def _segment_egg_multi_strategy(img):
    """通用分割（数据集图片专用）：轮廓填充 → Otsu → GrabCut → 腐蚀 → Canny。
    优先使用 flood-fill 专门处理白线黑底的轮廓图。
    对左下角线条较细的情况比 Otsu 更鲁棒。
    Returns (clean_mask, largest_contour, strategy_name, warning).
    """
    strategies = [
        ('轮廓填充', _segment_by_contour_floodfill),
        ('灰度Otsu', _segment_by_grayscale_otsu),
        ('GrabCut中心', _segment_by_grabcut_center),
        ('腐蚀法', _segment_by_binary_erosion),
        ('Canny', _segment_by_edge),
    ]
    for name, func in strategies:
        try:
            mask, contour, ok = func(img)
            if ok:
                quality = _evaluate_mask_quality(mask)
                # 轮廓填充质量极高且稳定 → 直接信任
                # GrabCut 需要更高阈值（0.3+）避免截断
                min_quality = 0.1 if name == '轮廓填充' else (0.1 if name == '灰度Otsu' else 0.3)
                if quality > min_quality:
                    return mask, contour, name, None
        except Exception:
            continue
    return None, None, None, '所有分割策略均失败'


def _segment_egg_photo(img):
    """照片分割（用户上传专用）：多算法竞争——边缘法 → K-means → GrabCut，取最优。
    
    不依赖单一算法，而是并行尝试多种不同原理的分割方法：
    - 边缘检测法：基于硬边缘，不受阴影影响（阴影是软边缘）
    - K-means聚类：基于颜色
    - GrabCut：基于颜色分布模型
    - Canny后补：最后手段
    
    每种方法独立评估质量，返回最优结果。
    Returns (clean_mask, largest_contour, strategy_name, warning).
    """
    results = []
    
    # 策略列表：(名称, 函数, 最低质量要求)
    candidates = [
        ('U2Net AI', _segment_by_u2net, 0.1),
        ('边缘填充', _segment_by_edge_floodfill, 0.1),
        ('K-means', _segment_by_kmeans_photo, 0.1),
        ('L*a*b*', _segment_by_lab, 0.1),
        ('边缘GrabCut', _segment_by_edge_guided_grabcut, 0.3),
        ('GrabCut中心', _segment_by_grabcut_center, 0.3),
        ('灰度Otsu', _segment_by_grayscale_otsu, 0.1),
        ('Canny', _segment_by_edge, 0.1),
    ]
    
    for name, func, min_qual in candidates:
        try:
            mask, contour, ok = func(img)
            if ok:
                quality = _evaluate_mask_quality(mask)
                if quality > min_qual:
                    results.append((quality, name, mask, contour))
        except Exception:
            continue
    
    if not results:
        return None, None, None, '所有分割策略均失败'
    
    # 按质量降序排列，取最优
    results.sort(key=lambda x: x[0], reverse=True)
    best = results[0]
    return best[2], best[3], best[1], None


# ═══════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════

def extract_features_from_image(image_path):
    """从鸡蛋图像文件中提取 19 个静态特征。"""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    mask, contour, _, _ = _segment_egg_multi_strategy(img)
    if mask is None or contour is None:
        return None
    return _compute_features(mask, contour)


def process_uploaded_image(uploaded_file):
    """上传照片全流程处理。"""
    result = {'success': False, 'steps': {}, 'features': None,
              'error': '', 'warning': None, 'strategy': None}
    try:
        if isinstance(uploaded_file, str):
            img = cv2.imread(uploaded_file)
        else:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            result['error'] = '无法读取图像文件'
            return result

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result['steps']['original'] = img_rgb
        result['steps']['grayscale'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 上传照片 → 使用 L*a*b* 照片专用链
        mask, contour, strategy, error = _segment_egg_photo(img)
        result['strategy'] = strategy
        if mask is None or contour is None:
            result['error'] = error or '未能在图像中检测到鸡蛋区域'
            return result

        quality = _evaluate_mask_quality(mask)
        if quality < 0.3:
            result['warning'] = (
                f'分割质量较低（{quality:.2f}）'
            )
        result['steps']['hsv_mask'] = mask

        # 轮廓平滑（仅用于可视化，不改变特征计算的 contour）
        epsilon = 0.005 * cv2.arcLength(contour, closed=True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        contour_viz = img_rgb.copy()
        # 先叠加 Canny 边缘（灰色细线，显示算法检测到的真实边缘）
        gray_viz = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges_viz = cv2.Canny(cv2.GaussianBlur(gray_viz, (5, 5), 0), 30, 100)
        overlay = contour_viz.copy()
        overlay[edges_viz > 0] = [100, 255, 100]  # 绿色边缘
        contour_viz = cv2.addWeighted(overlay, 0.3, contour_viz, 0.7, 0)

        # 绘制平滑后的轮廓
        cv2.drawContours(contour_viz, [smooth_contour], -1, (255, 0, 0), 3)
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(contour_viz, (cx, cy), 8, (255, 0, 0), -1)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_viz, (x, y), (x + w, y + h), (255, 255, 0), 3)
        result['steps']['contour_viz'] = contour_viz

        features = _compute_features(mask, contour)
        if features is None:
            result['error'] = '特征提取失败'
            return result
        result['features'] = features
        result['contour_image'] = mask
        result['success'] = True
    except Exception as e:
        result['error'] = f'处理出错: {str(e)}'
    return result


def _compute_features(mask, contour):
    """从二值掩膜和轮廓计算 19 个静态特征。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        return None
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()

    area = cv2.countNonZero(mask)
    perimeter = cv2.arcLength(contour, closed=True)
    cx = moments['m10'] / max(moments['m00'], 1)
    cy = moments['m01'] / max(moments['m00'], 1)

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, _), (ma, mi), _ = ellipse
        major = max(ma, mi)
        minor = min(ma, mi)
    else:
        major = np.sqrt(2 * (moments['mu20'] + moments['mu02'] + np.sqrt(
            (moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2
        )) / max(moments['m00'], 1))
        minor = np.sqrt(2 * (moments['mu20'] + moments['mu02'] - np.sqrt(
            (moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2
        )) / max(moments['m00'], 1))

    equiv_diameter = np.sqrt(4 * area / np.pi)
    eccentricity = np.sqrt(1 - (minor / max(major, 1))**2)
    shape_index = minor / max(major, 1) * 100
    circularity = 4 * np.pi * area / max(perimeter**2, 1)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(w * h, 1)

    cx_int = int(round(cx))
    left = mask[:, :cx_int]
    right = mask[:, cx_int:]
    la = cv2.countNonZero(left) if left.size > 0 else 0
    ra = cv2.countNonZero(right) if right.size > 0 else 0
    asym = abs(la - ra) / max(la + ra, 1)
    offset = abs(cx - (x + w / 2)) / max(w, 1)

    return {
        'Static_Area_像素面积': float(area),
        'Static_Perimeter_轮廓周长': float(perimeter),
        'Static_MajorAxisLength_长轴像素长度': float(major),
        'Static_MinorAxisLength_短轴像素长度': float(minor),
        'Static_EquivalentDiameter_等效圆直径': float(equiv_diameter),
        'Static_Eccentricity_离心率': float(eccentricity),
        'Static_ShapeIndex_机器视觉ESI': float(shape_index),
        'Static_Circularity_圆形度': float(circularity),
        'Static_Solidity_坚实度': float(solidity),
        'Static_Extent_延展度': float(extent),
        'Static_AsymmetryIndex_不对称指数': float(asym),
        'Static_MajorAxisOffsetRatio_长轴偏移率': float(offset),
        'Static_Hu1': float(hu[0]),
        'Static_Hu2': float(hu[1]),
        'Static_Hu3': float(hu[2]),
        'Static_Hu4': float(hu[3]),
        'Static_Hu5': float(hu[4]),
        'Static_Hu6': float(hu[5]),
        'Static_Hu7': float(hu[6]),
    }
