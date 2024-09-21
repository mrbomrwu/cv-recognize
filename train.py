import cv2
import numpy as np

class CircleDetector(object):
    def detectCircles(self, image, dp=1.2, minDist=50, param1=100, param2=30, minRadius=0, maxRadius=0):
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 中值模糊
        blurred = cv2.medianBlur(gray_image, 5)
        
        # 使用霍夫圆变换检测圆形
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        
        # 如果检测到圆形
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # print(f"Detected {len(circles)} circles")  # 调试输出
            regions = []
            for circle in circles:
                print(f"Circle: center=({circle[0]}, {circle[1]}), radius={circle[2]})")  # 调试输出
                
                # 提取圆形区域的ROI
                x, y, r = circle[0], circle[1], circle[2]
                x1, y1, x2, y2 = max(0, x-r), max(0, y-r), min(image.shape[1], x+r), min(image.shape[0], y+r)
                roi = image[y1:y2, x1:x2]
                regions.append(roi)
            return regions
        else:
            print("No circles detected")  # 调试输出
            return []

if __name__ == '__main__':
    src = 1  # 视频源，可以改为视频文件路径或0表示默认摄像头
    cap = cv2.VideoCapture(src)

    detector = CircleDetector()
    
    while True:
        if not cap.isOpened():
            print('相机未打开')
            break

        ret, frame = cap.read()

        if not ret:
            continue

        # 获取检测到的所有圆形区域
        regions = detector.detectCircles(frame, dp=1.2, minDist=50, param1=100, param2=45, minRadius=42, maxRadius=60)
        print(regions)

        # 将圆形区域展示或者进一步处理
        for idx, region in enumerate(regions):
            cv2.imshow(f'Region {idx}', region)

        # 显示原始图像
        cv2.imshow('image', frame)
        
        key = cv2.waitKey(int(1000/30)) & 0xFF
        if key == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
