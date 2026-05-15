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


def _segment_egg_multi_strategy(img):
    """多策略鸡蛋分割 (带质量评估 + 降级链).
    Returns (clean_mask, largest_contour, strategy_name, warning).
    """
    strategies = [
        ('GrabCut中心', _segment_by_grabcut_center),
        ('灰度Otsu', _segment_by_grayscale_otsu),
        ('腐蚀法', _segment_by_binary_erosion),
        ('L*a*b*', _segment_by_lab),
        ('EdgeGrabCut', _segment_by_edge_guided_grabcut),
        ('Canny', _segment_by_edge),
    ]

    for name, func in strategies:
        try:
            mask, contour, ok = func(img)
            if ok:
                quality = _evaluate_mask_quality(mask)
                if quality > 0.1:
                    return mask, contour, name, None
        except Exception:
            continue
    return None, None, None, '所有分割策略均失败'


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

        mask, contour, strategy, error = _segment_egg_multi_strategy(img)
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

        contour_viz = img_rgb.copy()
        cv2.drawContours(contour_viz, [contour], -1, (255, 0, 0), 3)
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
