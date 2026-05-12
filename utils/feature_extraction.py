"""Extract egg static features from images using OpenCV.
Port of MATLAB Code_1.m + segmentEggFromBlueBackground logic."""
import cv2
import numpy as np
import os


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

    # HSV segmentation (same pipeline as process_uploaded_image)
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

    binary = np.logical_and(mask_s, mask_v).astype(np.uint8) * 255

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    return _compute_features(mask, largest)


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
    """
    result = {'success': False, 'steps': {}, 'features': None, 'error': ''}
    
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
        result['steps']['original'] = img_rgb
        
        # Step 1: HSV Segmentation (port of MATLAB segmentEggFromBlueBackground)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1].astype(np.float32) / 255.0
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        
        # Saturation threshold: blue bg has high S, white egg has low S
        # MATLAB: graythresh(S) then maskS = S < max(0.10, min(0.60, tS * 1.05))
        # OpenCV equivalent using Otsu
        s_uint8 = (s_channel * 255).astype(np.uint8)
        tS = cv2.threshold(s_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        s_thresh = max(0.10, min(0.60, tS * 1.05))
        mask_s = s_channel < s_thresh
        
        # Value threshold: egg is usually brighter
        v_uint8 = (v_channel * 255).astype(np.uint8)
        tV = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        v_thresh = min(0.95, max(0.30, tV * 0.85))
        mask_v = v_channel > v_thresh
        
        # Combine masks
        binary_mask = (mask_s & mask_v).astype(np.uint8) * 255
        result['steps']['grayscale'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Clean up mask: remove small noise, fill holes
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep largest connected component
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            result['error'] = '未能在图像中检测到鸡蛋区域'
            return result
        
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        # Fill holes
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        
        result['steps']['hsv_mask'] = clean_mask
        
        # Create contour visualization (MATLAB style: red contour + centroid + bounding box)
        contour_viz = img_rgb.copy()
        cv2.drawContours(contour_viz, [largest_contour], -1, (255, 0, 0), 3)
        M = cv2.moments(clean_mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(contour_viz, (cx, cy), 8, (255, 0, 0), -1)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(contour_viz, (x, y), (x + w, y + h), (255, 255, 0), 3)
        result['steps']['contour_viz'] = contour_viz
        
        # Step 2: Compute features
        features = _compute_features(clean_mask, largest_contour)
        if features is None:
            result['error'] = '特征提取失败'
            return result
        
        result['features'] = features
        result['contour_image'] = clean_mask
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
