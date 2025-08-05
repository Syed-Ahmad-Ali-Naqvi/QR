import cv2
import numpy as np
import os

def estimate_dominant_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.rad2deg(theta)
        if 60 < angle < 120 or 150 < angle < 210:
            angles.append(angle - 90)
    return np.median(angles) if angles else 0

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def generate_candidate_angles(base_angle, margin=5, extra=None):
    if extra is None:
        extra = [90, 180, 270]
    base_angle = int(round(base_angle)) % 360
    nearby = [(base_angle + i) % 360 for i in range(-margin, margin + 1)]
    return sorted(set(nearby + extra))

def get_finder_patterns(qr_img, debug_img=None):
    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 300:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                squares.append(((cx, cy), area, approx.reshape(4, 2)))

    if debug_img is not None:
        for ((x, y), _, _) in squares:
            cv2.circle(debug_img, (x, y), 5, (0, 255, 255), -1)

    squares.sort(key=lambda x: x[1], reverse=True)
    return [s[2] for s in squares[:3]]  # Return top 3 square corners

def order_points(pts):
    # Order: top-left, top-right, bottom-right, bottom-left
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def average_box_corners(boxes):
    all_pts = np.concatenate(boxes)
    hull = cv2.convexHull(all_pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return order_points(box)

def warp_qr_image(img, box, size=500):
    dst_pts = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped

def align_qr_dynamic(image_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_angle = estimate_dominant_angle(img)
    print(f"🔧 Estimated base angle: {base_angle:.1f}°")
    candidates = generate_candidate_angles(base_angle)
    # temp = [-x for x in candidates] 
    print(f"🧪 Testing angles: {candidates}")
    # print(f"🧪 Testing angles(-): {temp}")
    
    best_result = None
    best_angle = None
    best_box = None
    # candidates.extend(temp)

    for angle in candidates:
        rotated = rotate_image(img, angle)
        debug_img = rotated.copy()
        squares = get_finder_patterns(rotated, debug_img=debug_img)

        if len(squares) >= 2:
            box = average_box_corners(squares)
            warped = warp_qr_image(rotated, box)
            score = len(squares)
            if best_result is None or score > len(best_box):
                best_result = warped
                best_angle = angle
                best_box = box
                cv2.imwrite(os.path.join(output_dir, f"debug_at_{angle:.0f}.png"), debug_img)

    if best_result is not None:
        output_path = os.path.join(output_dir, f"aligned_{best_angle:.0f}.png")
        cv2.imwrite(output_path, best_result)
        print(f"✅ Best result saved at {output_path} (angle: {best_angle:.1f}°)")
        return best_result, best_angle
    else:
        fallback_path = os.path.join(output_dir, "fallback_raw.png")
        cv2.imwrite(fallback_path, img)
        print(f"⚠️ Failed to align. Raw image saved at {fallback_path}")
        return img, 0


if __name__ == "__main__":
    for num in range(1,5):
        image = f'img{num}.jpg'
        out = f'res{num}'
        result, angle = align_qr_dynamic(image, output_dir=out)
        if result is not None:
            print(f"🎉 QR successfully aligned at {angle}°!")
        else:
            print("❌ Failed to align QR.")
