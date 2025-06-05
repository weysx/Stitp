import cv2
import numpy as np
import utlis


def preprocess(img):
    height = 1000
    width = 1000

    img = cv2.resize(img, (width, height))  # 将图像调整为预定的尺寸
    imgBlank = np.zeros((height, width, 3), np.uint8)  # 创建一个空白图像用于后续操作
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 使用高斯模糊平滑图像，减少噪声

    # 初始化阈值
    utlis.initializeTrackbars()

    while True:
        thres = utlis.valTrackbars()  # 获取阈值

        # 应用 Canny 边缘检测
        imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # 通过 Canny 边缘检测提取图像轮廓
        kernel = np.ones((5, 5))  # 创建一个 5x5 的矩阵，用于形态学操作
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # 使用膨胀操作增强图像中的边缘
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # 使用腐蚀操作减弱图像中的噪声

        # 复制图像用于轮廓检测
        imgContours = img.copy()  # 用于绘制所有检测到的轮廓
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测图像中的轮廓
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # 绘制所有检测到的轮廓

        cv2.imshow('Canny Edge Detection', imgContours)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break  # 确定阈值后按 'c' 继续执行

    # 查找最大轮廓并处理
    biggest, maxArea = utlis.biggestContour(contours)  # 找到面积最大的轮廓
    if biggest.size == 0:  # 如果没有找到最大轮廓
        print("error：没有找到最大轮廓。")
        exit()  # 退出程序

    biggest = utlis.reorder(biggest)  # 重排序最大轮廓的点，确保透视变换时顺序正确
    imgBigContour = img.copy()
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # 在图像上绘制出最大轮廓
    imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)  # 根据最大轮廓绘制矩形

    # 定义透视变换矩阵，将最大轮廓变换为指定尺寸
    pts1 = np.float32(biggest)  # 最大轮廓的四个顶点
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # 目标图像的四个顶点
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 计算透视变换矩阵
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))  # 应用透视变换，获得变换后的图像

    # 移除边界的 20 像素，获得更干净的图像
    imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored, (width, height))  # 重新调整图像大小

    # 转换为灰度图，并应用自适应阈值处理
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # 将变换后的图像转换为灰度图
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)  # 自适应阈值处理，提取图像中的细节
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)  # 反转图像的颜色
    imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)  # 使用中值滤波，进一步平滑图像

    # 图像数组以进行显示
    imageArray = ([img, imgGray, imgThreshold, imgContours],
                  [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    # 显示标签
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.75, lables)

    # cv2.imshow("process", stackedImage)  # 展示处理过程

    img_final = imgAdaptiveThre
    return img_final
