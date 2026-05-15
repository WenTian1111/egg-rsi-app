"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic.

Segmentation strategy (multi-layered fallback):
  PRIMARY:    L*a*b* + CLAHE + Otsu on b* channel (best for food/egg images)
  Fallback 1: Edge-Guided GrabCut (adaptive rect, handles off-center eggs)
  Fallback 2: HSV thresholding (original blue-bg fallback)
  Fallback 3: Canny edge detection (last resort)
"""
import cv2
import numpy as np
import os


# ─────────────────────────────────────────────
# PRIMARY: L*a*b* + CLAHE segmentation
# ─────────────────────────────────────────────

def _segment_by_lab(img):
    """Strategy A (PRIMARY): L*a*b* color space + CLAHE enhancement.

    L*a*b* separates color from intensity better than HSV for food images.
    The b* channel (yellow ↔ blue) is excellent for separating yellowish
    eggshell from typical backgrounds.
    CLAHE on the L channel handles uneven lighting robustly.
    """
    h, w = img.shape[:2]

    # 1. Convert to L*a*b*
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # 2. CLAHE on L channel → handles uneven lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    # 3. Merge back enhanced L with a, b
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    lab_eq_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    lab_eq_lab = cv2.cvtColor(lab_eq_rgb, cv2.COLOR_BGR2LAB)
    _, _, b_eq = cv2.split(lab_eq_lab)

    # 4. Gaussian blur on b* channel
    b_blur = cv2.GaussianBlur(b_eq, (7, 7), 0)

    # 5. Otsu thresholding on b* channel
    #    Eggs are yellowish → low b* values → invert if needed
    _, b_mask = cv2.threshold(
        b_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 6. Check which side of the threshold contains the egg
    #    Egg region tends to be less blue → lower b* → more likely 'background' in Otsu
    #    Check which region is more central/compact
    center_h, center_w = h // 2, w // 2
    center_roi = b_mask[center_h - h//6:center_h + h//6,
                        center_w - w//6:center_w + w//6]
    if center_roi.size > 0:
        center_fg_ratio = np.sum(center_roi > 0) / center_roi.size
        # If less than 30% of the center is "foreground", invert
        if center_fg_ratio < 0.3:
            b_mask = 255 - b_mask

    # 7. Morphological cleanup
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel_open)
    b_mask = cv2.morphologyEx(b_mask, cv2.MORPH_CLOSE, kernel_close)

    # 8. Fill holes
    b_mask = _fill_holes(b_mask)

    # 9. Find largest contour
    contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(b_mask)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return clean, largest, True


# ─────────────────────────────────────────────
# FALLBACK 1: Edge-Guided GrabCut
# ─────────────────────────────────────────────

def _segment_by_edge_guided_grabcut(img):
    """Strategy B: Edge-guided GrabCut with adaptive initialization.

    Instead of a fixed center rect (which fails for off-center eggs),
    uses edge detection to locate the largest foreground object,
    then initializes GrabCut with its bounding box.

    This handles eggs near edges, off-center, or in complex backgrounds.
    """
    h, w = img.shape[:2]

    # 1. Edge detection to find candidate regions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive threshold instead of Canny for more robust detection
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 4
    )
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Find the largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # 3. Expand rect slightly (15% margin), clamped to image bounds
    margin_x = max(10, int(cw * 0.15))
    margin_y = max(10, int(ch * 0.15))
    rx = max(0, x - margin_x)
    ry = max(0, y - margin_y)
    rw = min(w - rx, cw + 2 * margin_x)
    rh = min(h - ry, ch + 2 * margin_y)

    # Minimum rect size check
    if rw < 20 or rh < 20:
        return None, None, False

    rect = (rx, ry, rw, rh)

    # 4. Run GrabCut with the adaptive rect
    mask_gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask_gc, rect, bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)

    # 5. Post-process: keep only the most likely foreground region
    bin_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.uint8) * 255
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bin_mask = _fill_holes(bin_mask)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(bin_mask)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean, largest, True


# ─────────────────────────────────────────────
# FALLBACK 2: HSV thresholding
# ─────────────────────────────────────────────

def _segment_by_hsv(img):
    """Strategy C: HSV threshold segmentation (fallback for blue-bg images)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    s_u = (s * 255).astype(np.uint8)
    tS = cv2.threshold(s_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    s_th = max(0.10, min(0.60, tS * 1.05))

    v_u = (v * 255).astype(np.uint8)
    tV = cv2.threshold(v_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    v_th = min(0.95, max(0.30, tV * 0.85))

    binary = ((s < s_th) & (v > v_th)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(binary)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return clean, largest, True


# ─────────────────────────────────────────────
# FALLBACK 3: Canny edge detection
# ─────────────────────────────────────────────

def _segment_by_edge(img):
    """Strategy D: Canny edge detection (last resort)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask, largest, True


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _fill_holes(mask):
    """Fill small holes in a binary mask using floodfill."""
    h, w = mask.shape
    mask_ext = np.zeros((h + 2, w + 2), np.uint8)
    mask_ext[1:-1, 1:-1] = mask
    cv2.floodFill(mask_ext, None, (0, 0), 255)
    mask_inv = 255 - mask_ext[1:-1, 1:-1]
    return mask | mask_inv


def _evaluate_mask_quality(mask):
    """Check if mask is plausible (egg-like). Returns quality score 0-1."""
    area = cv2.countNonZero(mask)
    h, w = mask.shape
    total = h * w
    ratio = area / total if total > 0 else 0
    if ratio < 0.01 or ratio > 0.95:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area_contour = cv2.contourArea(largest)
    if area_contour < 1:
        return 0.0

    # Solidity test
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = min(1.0, area / max(hull_area, 1))

    # Compactness test: a good egg should have reasonable circularity
    perimeter = cv2.arcLength(largest, closed=True)
    circularity = 4 * np.pi * area / max(perimeter**2, 1) if perimeter > 0 else 0

    # Combined score
    score = min(1.0, (solidity * 0.6 + (1 - abs(circularity - 0.5)) * 0.4))

    if score < 0.2:
        return 0.0
    return score


def _segment_egg_multi_strategy(img):
    """Multi-strategy egg segmentation with fallback chain.

    PRIMARY:    L*a*b* + CLAHE (best for food/egg images, handles uneven lighting)
    Fallback 1: Edge-Guided GrabCut (handles off-center/complex backgrounds)
    Fallback 2: HSV threshold (blue background images)
    Fallback 3: Canny edge detection (last resort)
    Returns (clean_mask, largest_contour, strategy_name, warning).
    """
    strategies = [
        ('L*a*b*', _segment_by_lab),
        ('Edge+GrabCut', _segment_by_edge_guided_grabcut),
        ('HSV', _segment_by_hsv),
        ('Edge', _segment_by_edge),
    ]

    for name, func in strategies:
        mask, contour, ok = func(img)
        if ok:
            quality = _evaluate_mask_quality(mask)
            if quality > 0.1:
                return mask, contour, name, None

    return None, None, None, '所有分割策略均失败'


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def extract_features_from_image(image_path):
    """Extract the 19 static features from an egg image."""
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
    """Full pipeline for uploaded egg photos.

    Returns dict with:
      - 'steps': dict of intermediate images
      - 'contour_image': the extracted contour image (numpy array)
      - 'features': dict of 19 static features
      - 'success': bool
      - 'error': error message if failed
      - 'warning': warning message if segmentation quality is suspect
      - 'strategy': the segmentation strategy that succeeded
    """
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
                f'分割质量较低（{quality:.2f}），可能影响特征准确性，'
                f'建议使用蓝色背景或高对比度图片'
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
    """Compute the 19 static features from a clean binary mask and its contour."""
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
