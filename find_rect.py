import cv2
import numpy as np

def preprocess_image(img):
    # 灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值模糊
    blurred = cv2.medianBlur(gray, 5)
    # Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges

def detect_largest_rectangle(img, edges):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    largest_rectangle = None
    for cnt in contours:
        # 轮廓近似
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 过滤非矩形轮廓
        if len(approx) == 4:
            # 面积过滤
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_rectangle = approx
    if largest_rectangle is not None:
        cv2.drawContours(img, [largest_rectangle], -1, (0, 255, 0), 2)
    return largest_rectangle

cap = cv2.VideoCapture(1)
cv2.namedWindow('rect', cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        if img is not None:
            edges = preprocess_image(img)
            largest_rectangle = detect_largest_rectangle(img, edges)
            cv2.imshow("rect", img)
            cv2.imshow('edges', edges)
            if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
                break

cap.release()
cv2.destroyAllWindows()
