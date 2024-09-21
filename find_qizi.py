import cv2 as cv
import numpy as np


def nothing(x):
    pass


def morphological_operation(frame):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 获取图像结构化元素
    dst = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)  # 闭操作
    return dst


def color_detetc(frame):

    hmin1 = cv.getTrackbarPos('hmin1', 'color_adjust1')
    hmax1 = cv.getTrackbarPos('hmax1', 'color_adjust1')
    smin1 = cv.getTrackbarPos('smin1', 'color_adjust1')
    smax1 = cv.getTrackbarPos('smax1', 'color_adjust1')
    vmin1 = cv.getTrackbarPos('vmin1', 'color_adjust1')
    vmax1 = cv.getTrackbarPos('vmax1', 'color_adjust1')

    hmin2 = cv.getTrackbarPos('hmin2', 'color_adjust2')
    hmax2 = cv.getTrackbarPos('hmax2', 'color_adjust2')
    smin2 = cv.getTrackbarPos('smin2', 'color_adjust2')
    smax2 = cv.getTrackbarPos('smax2', 'color_adjust2')
    vmin2 = cv.getTrackbarPos('vmin2', 'color_adjust2')
    vmax2 = cv.getTrackbarPos('vmax2', 'color_adjust2')
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # hsv 色彩空间 分割肤色
    lower_hsv1 = np.array([hmin1, smin1, vmin1])
    upper_hsv1 = np.array([hmax1, smax1, vmax1])
    print(lower_hsv1, upper_hsv1)
    mask1 = cv.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)  # hsv 掩码
    lower_hsv2 = np.array([hmin2, smin2, vmin2])
    upper_hsv2 = np.array([hmax2, smax2, vmax2])
    # print(lower_hsv2, upper_hsv2)
    mask2 = cv.inRange(hsv, lowerb=lower_hsv2, upperb=upper_hsv2)  # hsv 掩码
    ret, thresh1 = cv.threshold(mask1, 40, 255, cv.THRESH_BINARY)  # 二值化处理
    ret, thresh2 = cv.threshold(mask2, 40, 255, cv.THRESH_BINARY)  # 二值化处理

    return thresh1, thresh2


def main():
    cv.namedWindow("color_adjust1")
    cv.namedWindow("color_adjust2")
    cv.createTrackbar("hmin1", "color_adjust1", 0, 179, nothing)
    cv.createTrackbar("hmax1", "color_adjust1", 179, 179, nothing)
    cv.createTrackbar("smin1", "color_adjust1", 0, 255, nothing)
    cv.createTrackbar("smax1", "color_adjust1", 255, 255, nothing)
    cv.createTrackbar("vmin1", "color_adjust1", 0, 255, nothing)
    cv.createTrackbar("vmax1", "color_adjust1", 50, 255, nothing)

    cv.createTrackbar("hmin2", "color_adjust2", 0, 179, nothing)
    cv.createTrackbar("hmax2", "color_adjust2", 179, 179, nothing)
    cv.createTrackbar("smin2", "color_adjust2", 0, 255, nothing)
    cv.createTrackbar("smax2", "color_adjust2", 30, 255, nothing)
    cv.createTrackbar("vmin2", "color_adjust2", 200, 255, nothing)
    cv.createTrackbar("vmax2", "color_adjust2", 255, 255, nothing)

    capture = cv.VideoCapture(1)  # 打开电脑自带摄像头，如果参数是1会打开外接摄像头

    while True:
        ret, frame = capture.read()
        mask1, mask2 = color_detetc(frame)
        scr1 = morphological_operation(mask1)
        scr2 = morphological_operation(mask2)

        contours1, heriachy1 = cv.findContours(scr1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓点集(坐标)
        contours2, heriachy2 = cv.findContours(scr2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓点集(坐标)
        cv.drawContours(frame, contours1, -1, (0, 0, 255), 2)
        cv.drawContours(frame, contours2, -1, (0, 255, 0), 2)
        for i, contour in enumerate(contours1):
            area1 = cv.contourArea(contour)
            if area1 > 20:
                (x1, y1), radius1 = cv.minEnclosingCircle(contours1[i])
                x1 = int(x1)
                y1 = int(y1)
                center1 = (int(x1), int(y1))
                radius1 = int(radius1)
                cv.circle(frame, center1, 3, (0, 0, 255), -1)  # 画出重心
                # print("黑色棋子:", (x1, y1))
                cv.putText(frame, "black:", (x1, y1), cv.FONT_HERSHEY_SIMPLEX,
                       1, [255, 255, 255])
        for k, contour in enumerate(contours2):
            area2 = cv.contourArea(contour)
            if area2 > 20:
                (x2, y2), radius2 = cv.minEnclosingCircle(contours2[k])
                x2 = int(x2)
                y2 = int(y2)
                center2 = (int(x2), int(y2))
                radius2 = int(radius2)
                cv.circle(frame, center2, 3, (0, 0, 255), -1)  # 画出重心
                # print("白色棋子:", (x2, y2))
                cv.putText(frame, "white:", (x2, y2), cv.FONT_HERSHEY_SIMPLEX,
                       1, [255, 255, 255])
        cv.imshow("mask1", mask1)
        cv.imshow("mask2", mask2)
        cv.imshow("frame", frame)
        c = cv.waitKey(50)
        if c == 27:
            break


main()

cv.waitKey(0)
cv.destroyAllWindows()

