"""Debug segmentation on egg contour images"""
import cv2
import numpy as np

img_path = 'data/egg_images/1号鸡蛋轮廓.jpg'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]

# Otsu threshold
thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Otsu threshold value: {thresh_val}')

# Corner check (as done in generate_pipeline_images)
corner_pixels = [binary[5, 5], binary[5, w-5], binary[h-5, 5], binary[h-5, w-5]]
print(f'Corner pixels: {corner_pixels}')
should_invert = sum(p > 127 for p in corner_pixels) >= 2
print(f'Should invert: {should_invert}')

# Pixels cut off - values below Otsu threshold but above 0
low_vals = gray[(gray > 0) & (gray < thresh_val)]
print(f'Pixels 0 < gray < {thresh_val}: {len(low_vals)} ({len(low_vals)/gray.size*100:.2f}%)')
if len(low_vals) > 0:
    print(f'  Range: {low_vals.min()} - {int(low_vals.max())}, mean: {low_vals.mean():.1f}')

# Fix approach: use a lower threshold (e.g. 30) for contour images
_, binary_fixed = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
contours_fixed, _ = cv2.findContours(binary_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_fixed:
    largest_fixed = max(contours_fixed, key=cv2.contourArea)
    xf, yf, wf, hf = cv2.boundingRect(largest_fixed)
    print(f'\nFixed threshold (30) bbox: x={xf}, y={yf}, w={wf}, h={hf}')
    area_fixed = cv2.contourArea(largest_fixed)
    print(f'Fixed contour area: {area_fixed} ({area_fixed/(h*w)*100:.1f}%)')

# Compare with Otsu
contours_otsu, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_otsu:
    largest_otsu = max(contours_otsu, key=cv2.contourArea)
    xo, yo, wo, ho = cv2.boundingRect(largest_otsu)
    print(f'Otsu bbox: x={xo}, y={yo}, w={wo}, h={ho}')
    area_otsu = cv2.contourArea(largest_otsu)
    print(f'Otsu contour area: {area_otsu} ({area_otsu/(h*w)*100:.1f}%)')
    print(f'Fixed vs Otsu: Fixed extends {yf-yo}px more top, {xf-xo}px more left')

# Test across multiple eggs
print('\n=== Testing across eggs 1, 10, 20, 30, 40 ===')
for eid in [1, 10, 20, 30, 40]:
    p = f'data/egg_images/{eid}号鸡蛋轮廓.jpg'
    gi = cv2.imread(p)
    if gi is None:
        print(f'Egg {eid}: cannot read')
        continue
    gy = cv2.cvtColor(gi, cv2.COLOR_BGR2GRAY)
    tv, bi = cv2.threshold(gy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bf = cv2.threshold(gy, 30, 255, cv2.THRESH_BINARY)
    co, _ = cv2.findContours(bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cf, _ = cv2.findContours(bf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ao = cv2.contourArea(max(co, key=cv2.contourArea)) if co else 0
    af = cv2.contourArea(max(cf, key=cv2.contourArea)) if cf else 0
    diff = (af - ao) / ao * 100 if ao > 0 else 0
    print(f'Egg {eid}: Otsu={tv}, OtsuArea={ao/(gy.size)*100:.1f}%, FixedArea={af/(gy.size)*100:.1f}%, Diff={diff:+.1f}%')
