import cv2
import numpy as np
import matplotlib.pyplot as plt

#这部分是Chatgpt给的判断方向的方法（不可行）
def detect_arrow_direction(contour, image):

    # 计算凸包（确保箭头完整）
    hull = cv2.convexHull(contour)

    # 逼近多边形，找出关键拐点
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(approx)

    # 计算箭头的关键点
    left_most = tuple(contour[contour[:, :, 0].argmin()][0])  # 最左点
    right_most = tuple(contour[contour[:, :, 0].argmax()][0])  # 最右点
    top_most = tuple(contour[contour[:, :, 1].argmin()][0])  # 最上点
    bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])  # 最下点

    # 计算重心（质心）
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return "Unknown"
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # **可视化关键点**
    cv2.circle(image, left_most, 5, (0, 255, 0), -1)  # 绿色：最左点
    cv2.circle(image, right_most, 5, (255, 0, 0), -1)  # 蓝色：最右点
    cv2.circle(image, top_most, 5, (0, 0, 255), -1)  # 红色：最上点
    cv2.circle(image, bottom_most, 5, (255, 255, 0), -1)  # 青色：最下点
    cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)  # 黄色：重心

    # **打印调试信息**
    print(f"🔍 关键点 - 左: {left_most}, 右: {right_most}, 上: {top_most}, 下: {bottom_most}, 质心: ({cx}, {cy})")

    # **方向判定逻辑**
    if cx < left_most[0]:  # 箭头头部在左侧
        return "Left"
    elif cx > right_most[0]:  # 箭头头部在右侧
        return "Right"
    elif cy < top_most[1]:  # 箭头头部在上方
        return "Forward"

    return "Unknown"  # 无法识别方向

# 边缘检测，灰度，噪声，输出原图和灰度图还有判断结果（main code）
def trackArrows_CV(image_path):
    img = cv2.imread(image_path)

    # preconditioning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Noise reduction
    edges = cv2.Canny(blurred, 50, 150)  # Canny
    # Edge detect(test)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = img[y:y + h, x:x + w] # ROI extraction

    # perspective transformation（接上面方向检测）
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])  # 原始 ROI 角点
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # 目标变换后的角点
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 计算变换矩阵
    roi_warped = cv2.warpPerspective(img, matrix, (w, h))  # 透视变换
    # detect arrow_direction
    arrow_direction = detect_arrow_direction(largest_contour, img)



    # result of edge detection
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Picture")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title("ROI Extraction")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(roi_warped, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Perspective Transformation")
    axes[2].axis("off")

    plt.suptitle(f"Arrow Direction: {arrow_direction}")  # 这里改成箭头方向
    plt.show()

    print(f"箭头方向: {arrow_direction}")

image_paths = [
    "/Users/z.s.r/Desktop/assignment2/picture/right.jpg",
    "/Users/z.s.r/Desktop/assignment2/picture/forward.jpg",
    "/Users/z.s.r/Desktop/assignment2/picture/left.jpg"
]

for path in image_paths:
    trackArrows_CV(path)