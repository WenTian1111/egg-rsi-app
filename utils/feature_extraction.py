"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic.
V2: Two-stage segmentation pipeline for real-world robustness."""

import cv2
import numpy as np
import os


# ─── Blue background HSV range (OpenCV: H 0-179, S 0-255, V 0-255) ───
BLUE_HUE_LOW = 90
BLUE_HUE_HIGH = 130


def _detect_blue_background(hsv):
    """Check if the image has a significant blue background region."""
    h_channel = hsv[:, :, 0].astype(np.float32)
    # Proportion of pixels in blue hue range
    blue_mask = (h_channel >= BLUE_HUE_LOW) & (h_channel <= BLUE_HUE_HIGH)
    blue_ratio = np.sum(blue_mask) / blue_mask.size
    return blue_ratio > 0.25  # at least 25% of image is blue


def _hsv_segmentation(img):
    """
    Stage 1: HSV-based blue background removal.
    Returns (mask, success_flag).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0

    # ── Method A: Blue-hue background removal ──
    # If blue background is detected, use exact blue hue range
    if _detect_blue_background(hsv):
        # Blue background mask
        bg_h = (h_channel >= BLUE_HUE_LOW) & (h_channel <= BLUE_HUE_HIGH)
        # Also require moderate saturation (blue bg is usually saturated)
        bg_s = s_channel > 0.15
        # Also require moderate value (not too dark)
        bg_v = v_channel > 0.15  # but not too bright white
        bg_mask = bg_h & bg_s & bg_v
        # Invert: egg = not background
        egg_mask = (~bg_mask).astype(np.uint8) * 255
    else:
        # ── Method B: No blue background detected ──
        # Fall back to original S+V Otsu approach,
        # but with adaptive thresholds
        s_uint8 = (s_channel * 255).astype(np.uint8)
        tS = cv2.threshold(s_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        s_thresh = max(0.10, min(0.60, tS * 1.05))
        mask_s = s_channel < s_thresh  # egg has low saturation

        v_uint8 = (v_channel * 255).astype(np.uint8)
        tV = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        v_thresh = min(0.95, max(0.30, tV * 0.85))
        mask_v = v_channel > v_thresh  # egg is usually brighter

        egg_mask = (mask_s & mask_v).astype(np.uint8) * 255

    return egg_mask


def _edge_segmentation(img):
    """
    Stage 2: Edge-based segmentation (fallback for non-blue backgrounds).
    Uses Canny edge detection + contour filling.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Adaptive Canny thresholds using median
    med = np.median(blurred)
    low = max(10, int(0.5 * med))
    high = min(255, int(1.5 * med))
    edges = cv2.Canny(blurred, low, high)

    # Dilate to close edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill holes by finding the largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask = np.zeros_like(gray)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    # Close remaining holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    return mask


def _clean_mask(binary_mask):
    """Clean a binary mask: remove noise, fill holes, keep largest component."""
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(cleaned)
    cv2.drawContours(final_mask, [largest], -1, 255, -1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    return final_mask, largest


def _assess_mask_quality(mask, img_shape):
    """Check if the segmented mask is reasonable."""
    if mask is None:
        return False
    area = cv2.countNonZero(mask)
    total = img_shape[0] * img_shape[1]
    ratio = area / total
    # Mask should cover between 5% and 70% of the image
    return 0.05 < ratio < 0.70


def segment_egg(img):
    """
    Two-stage egg segmentation pipeline:
    1. HSV-based segmentation (works well for blue/dark backgrounds)
    2. Edge-based segmentation fallback (general purpose)
    
    Returns (mask, largest_contour, method_used, debug_images).
    """
    h, w = img.shape[:2]
    debug = {}

    # Stage 1: HSV segmentation
    hsv_mask = _hsv_segmentation(img)
    debug['hsv_stage'] = hsv_mask
    mask1, contour1 = _clean_mask(hsv_mask)

    if _assess_mask_quality(mask1, img.shape):
        return mask1, contour1, 'hsv', debug

    # Stage 2: Edge-based segmentation (fallback)
    edge_mask = _edge_segmentation(img)
    debug['edge_stage'] = edge_mask
    mask2, contour2 = _clean_mask(edge_mask) if edge_mask is not None else (None, None)

    if _assess_mask_quality(mask2, img.shape):
        return mask2, contour2, 'edge', debug

    # If both fail, return the better of the two
    if mask1 is not None and mask2 is not None:
        area1 = cv2.countNonZero(mask1)
        area2 = cv2.countNonZero(mask2)
        return (mask1, contour1, 'hsv', debug) if area1 > area2 else (mask2, contour2, 'edge', debug)
    elif mask1 is not None:
        return mask1, contour1, 'hsv', debug
    elif mask2 is not None:
        return mask2, contour2, 'edge', debug
    else:
        return None, None, 'failed', debug


def extract_features_from_image(image_path):
    """Extract the 19 static features from an egg image file."""
    if not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    mask, contour, method, _ = segment_egg(img)
    if mask is None or contour is None:
        return None

    return _compute_features(mask, contour)


def process_uploaded_image(uploaded_file):
    """
    Full pipeline for uploaded egg photos.
    Two-stage segmentation: HSV → Edge fallback.
    Returns dict with steps, features, and debug info.
    """
    result = {'success': False, 'steps': {}, 'features': None, 'error': '', 'method': ''}

    try:
        # Read image
        if isinstance(uploaded_file, str):
            img = cv2.imread(uploaded_file)
        else:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            result['error'] = '无法读取图像文件'
            return result

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 1: Original
        result['steps']['original'] = img_rgb

        # Step 2: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result['steps']['grayscale'] = gray

        # Step 3: Two-stage segmentation
        mask, largest_contour, method, debug = segment_egg(img)
        result['method'] = method

        if mask is None or largest_contour is None:
            result['error'] = '未能在图像中检测到鸡蛋区域（两阶段分割均失败）'
            return result

        result['steps']['hsv_mask'] = mask

        # Step 4: Contour visualization (red contour + blue centroid + yellow bounding box)
        contour_viz = img_rgb.copy()
        cv2.drawContours(contour_viz, [largest_contour], -1, (255, 0, 0), 3)
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(contour_viz, (cx, cy), 8, (255, 0, 0), -1)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(contour_viz, (x, y), (x + w, y + h), (255, 255, 0), 3)
        result['steps']['contour_viz'] = contour_viz

        # Compute features
        features = _compute_features(mask, largest_contour)
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
    # NOTE: Raw Hu moments (no log-transform), matching MATLAB Code_1.m and training data.

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
