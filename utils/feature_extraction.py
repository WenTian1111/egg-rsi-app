"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic.
V3: Three-stage segmentation with CLAHE normalization + GrabCut fallback."""

import cv2
import numpy as np
import os


# ─── Blue background HSV range (OpenCV: H 0-179, S 0-255, V 0-255) ───
BLUE_HUE_LOW = 90
BLUE_HUE_HIGH = 130


def _apply_clahe(gray):
    """Apply CLAHE to normalize uneven lighting on a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _detect_blue_background(hsv):
    """Check if the image has a significant blue background region."""
    h_channel = hsv[:, :, 0].astype(np.float32)
    blue_mask = (h_channel >= BLUE_HUE_LOW) & (h_channel <= BLUE_HUE_HIGH)
    blue_ratio = np.sum(blue_mask) / blue_mask.size
    return blue_ratio > 0.25


def _hsv_segmentation(img):
    """
    Stage 1: HSV-based segmentation.
    Method A (blue bg): exact blue hue range → invert.
    Method B (other): adaptive S+V Otsu thresholds with CLAHE-normalized V.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0

    # ── Method A: Blue-hue background removal ──
    if _detect_blue_background(hsv):
        bg_h = (h_channel >= BLUE_HUE_LOW) & (h_channel <= BLUE_HUE_HIGH)
        bg_s = s_channel > 0.15
        bg_v = v_channel > 0.15
        egg_mask = (~(bg_h & bg_s & bg_v)).astype(np.uint8) * 255
        return egg_mask

    # ── Method B: No blue background ──
    # Use CLAHE-normalized grayscale for better shadow handling
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = _apply_clahe(gray)

    # Adaptive thresholding (handles uneven lighting)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=5
    )

    # The egg is usually the darker object on a lighter background
    # or lighter object on darker background → auto-detect
    corner_pixels = [binary[5, 5], binary[5, -5], binary[-5, 5], binary[-5, -5]]
    if sum(p > 127 for p in corner_pixels) >= 2:
        binary = cv2.bitwise_not(binary)

    # Fall back to Otsu if adaptive threshold produces garbage
    # (too much noise → too many contours)
    contours_check, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_check) > 30:  # noisy result
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corner = [binary[5, 5], binary[5, -5], binary[-5, 5], binary[-5, -5]]
        if sum(p > 127 for p in corner) >= 2:
            binary = cv2.bitwise_not(binary)

    return binary


def _edge_segmentation(img):
    """
    Stage 2: Edge-based segmentation (fallback for non-blue backgrounds
    or when HSV fails). Uses CLAHE + Canny for better edge detection
    under uneven lighting.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize lighting
    norm = _apply_clahe(gray)
    blurred = cv2.GaussianBlur(norm, (7, 7), 1.5)

    # Compute adaptive Canny thresholds based on image statistics
    med = np.median(blurred)
    low = max(10, int(0.3 * med))
    high = min(255, int(1.2 * med))
    edges = cv2.Canny(blurred, low, high)

    # Aggressive closing to ensure egg boundary is a closed loop
    ksize = max(5, min(15, w // 40, h // 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes by finding the largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask = np.zeros_like(gray)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    # Final close to fill any remaining holes
    fill_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, fill_k)

    return mask


def _grabcut_segmentation(img):
    """
    Stage 3: GrabCut segmentation (ultimate fallback for complex backgrounds).
    Requires manual rectangle initialization based on image center.
    """
    h, w = img.shape[:2]
    # Place an initial rectangle covering the central 80% of the image
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # GC_FGD = 1, GC_PR_FGD = 3 → treat both as foreground
    result_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)

    return result_mask


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
    return 0.03 < ratio < 0.80  # slightly more permissive than V2


def segment_egg(img):
    """
    Three-stage egg segmentation pipeline:
    1. HSV-based segmentation (works well for blue/dark backgrounds)
    2. Edge-based segmentation (general purpose, CLAHE normalized)
    3. GrabCut (ultimate fallback for complex backgrounds)
    
    Returns (mask, largest_contour, method_used, debug_images).
    """
    debug = {}
    results = []

    # Stage 1: HSV segmentation
    hsv_mask = _hsv_segmentation(img)
    debug['hsv'] = hsv_mask
    mask1, contour1 = _clean_mask(hsv_mask)
    results.append((mask1, contour1, 'hsv', _assess_mask_quality(mask1, img.shape)))

    # Stage 2: Edge-based segmentation (if needed)
    if not any(q for _, _, _, q in results):
        edge_mask = _edge_segmentation(img)
        debug['edge'] = edge_mask
        mask2, contour2 = _clean_mask(edge_mask) if edge_mask is not None else (None, None)
        results.append((mask2, contour2, 'edge', _assess_mask_quality(mask2, img.shape)))

    # Stage 3: GrabCut (if everything above failed)
    if not any(q for _, _, _, q in results):
        grabcut_mask = _grabcut_segmentation(img)
        debug['grabcut'] = grabcut_mask
        mask3, contour3 = _clean_mask(grabcut_mask)
        results.append((mask3, contour3, 'grabcut', _assess_mask_quality(mask3, img.shape)))

    # Return the best result
    for mask, contour, method, quality in results:
        if quality and mask is not None and contour is not None:
            return mask, contour, method, debug

    # Absolute fallback: return the one with largest area
    best = max(results, key=lambda r: cv2.countNonZero(r[0]) if r[0] is not None else 0)
    mask, contour, method = best[0], best[1], best[2]
    return mask, contour, method, debug


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
    Three-stage segmentation: HSV → Edge → GrabCut.
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

        # Step 2: Grayscale + CLAHE-normalized (for display)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm = _apply_clahe(gray)
        result['steps']['grayscale'] = norm

        # Step 3: Three-stage segmentation
        mask, largest_contour, method, debug = segment_egg(img)
        result['method'] = method

        if mask is None or largest_contour is None:
            result['error'] = '未能在图像中检测到鸡蛋区域（三阶段分割均失败）'
            return result

        result['steps']['hsv_mask'] = mask

        # Step 4: Contour visualization
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
