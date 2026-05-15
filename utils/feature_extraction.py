"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic."""
import cv2
import numpy as np
import os


def _segment_by_hsv(img):
    """Strategy A: HSV threshold segmentation for blue background images.
    Returns (mask, success) where mask is binary mask uint8 or None on failure.
    """
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

    binary = (mask_s & mask_v).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return clean_mask, True


def _segment_by_edge(img):
    """Strategy B: Edge detection + contour filling for non-blue backgrounds.
    Uses Canny edge detection, dilation, and flood fill to find egg contour.
    Returns (mask, largest_contour, success).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask, largest, True


def _segment_by_adaptive(img):
    """Strategy C: Adaptive threshold for low-contrast images.
    Returns (mask, largest_contour, success).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 8)

    h, w = binary.shape
    corner_pixels = [binary[5, 5], binary[5, w-5], binary[h-5, 5], binary[h-5, w-5]]
    if sum(p > 127 for p in corner_pixels) >= 2:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, False

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return clean_mask, largest, True


def _evaluate_mask_quality(mask):
    """Check if mask is plausible (egg-like). Returns quality score 0-1."""
    area = cv2.countNonZero(mask)
    h, w = mask.shape
    total = h * w
    ratio = area / total if total > 0 else 0
    if ratio < 0.01 or ratio > 0.95:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area_contour = cv2.contourArea(largest)
    if area_contour < 1:
        return 0.0
    solidity = area / max(area_contour, 1)
    if solidity < 0.3:
        return 0.0
    return min(1.0, solidity * ratio * 100)


def _segment_egg_multi_strategy(img):
    """Multi-strategy egg segmentation with fallback chain.
    Returns (clean_mask, largest_contour, strategy_used, warning).
    """
    mask, success = _segment_by_hsv(img)
    if success:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            quality = _evaluate_mask_quality(mask)
            if quality > 0.1:
                return mask, max(contours, key=cv2.contourArea), 'HSV', None

    mask, contour, success = _segment_by_edge(img)
    if success:
        quality = _evaluate_mask_quality(mask)
        if quality > 0.1:
            return mask, contour, 'Edge', None

    mask, contour, success = _segment_by_adaptive(img)
    if success:
        return mask, contour, 'Adaptive', None

    return None, None, None, '所有分割策略均失败'


def extract_features_from_image(image_path):
    """
    Extract the 19 static features from an egg image.
    Uses the same HSV segmentation pipeline as process_uploaded_image()
    to ensure consistent features between training and inference.
    Returns dict of feature_name -> value, or None on failure.
    """
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
    """
    Full pipeline for uploaded egg photos.
    Takes an uploaded image file (BytesIO or file path), runs:
    Original → Grayscale → HSV Segmentation → Binary Mask → Contour → Features

    Returns dict with:
      - 'steps': dict of intermediate images (original_rgb, grayscale, hsv_mask, contour_viz)
      - 'contour_image': the extracted contour image (numpy array)
      - 'features': dict of 19 static features
      - 'success': bool
      - 'error': error message if failed
      - 'warning': warning message if segmentation quality is suspect
      - 'strategy': the segmentation strategy that succeeded
    """
    result = {'success': False, 'steps': {}, 'features': None, 'error': '', 'warning': None, 'strategy': None}

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
            result['warning'] = f'分割质量较低（{quality:.2f}），可能影响特征准确性，建议使用蓝色背景或高对比度图片'

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
    """
    Compute the 19 static features from a clean binary mask and its contour.
    Port of MATLAB Code_1.m logic.
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return None

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # NOTE: Do NOT log-transform Hu moments.
    # MATLAB Code_1.m saves raw Hu moments (no log).
    # The sklearn models were trained on raw Hu values from CSV.
    # Log-transform would produce a different feature distribution.

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
            (moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2)) / max(moments['m00'], 1))
        minor = np.sqrt(2 * (moments['mu20'] + moments['mu02'] - np.sqrt(
            (moments['mu20'] - moments['mu02'])**2 + 4 * moments['mu11']**2)) / max(moments['m00'], 1))

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
