import cv2
import numpy as np


def stackImages(imgArray, scale, lables=[]):
    """
    将多张图像按指定比例缩放后堆叠在一个窗口中进行显示，并添加标签。
    参数:
        imgArray: 包含图像的二维数组，每个元素可以是单张图像或图像列表。
        scale: 缩放比例，用于调整图像大小。
        lables: 标签列表，用于标记每张图像。
    返回:
        ver: 组合后的图像矩阵。
    """
    rows = len(imgArray)  # 获取图像行数
    cols = len(imgArray[0])  # 获取图像列数
    rowsAvailable = isinstance(imgArray[0], list)  # 检查 imgArray 中是否为嵌套列表
    width = imgArray[0][0].shape[1]  # 获取图像宽度
    height = imgArray[0][0].shape[0]  # 获取图像高度

    # 当 imgArray 是二维列表时，逐行和逐列缩放图像并将灰度图像转换为彩色
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)  # 缩放图像
                if len(imgArray[x][y].shape) == 2:  # 如果是灰度图像，则转换为 BGR 彩色图像
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)  # 创建一个空白图像用于填充
        hor = [imageBlank] * rows  # 水平拼接每一行图像
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])  # 将行内的图像水平拼接
        ver = np.vstack(hor)  # 将所有行垂直拼接
    else:
        # 如果 imgArray 是一维列表，逐个缩放图像并转换灰度图像
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)  # 缩放图像
            if len(imgArray[x].shape) == 2:  # 转换灰度图像为 BGR 彩色图像
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)  # 将一维列表内的图像水平拼接

    # 添加标签
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                # 绘制白色矩形框，用于显示标签
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                # 显示标签文字
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver


def reorder(myPoints):
    """
    对四个点进行排序，使其符合透视变换的顺序：左上、右上、左下、右下。
    参数:
        myPoints: 四个点的坐标，形状为 (4, 2)。
    返回:
        myPointsNew: 排序后的点集。
    """
    myPoints = myPoints.reshape((4, 2))  # 重塑为 4x2 矩阵
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)  # 初始化新点集
    add = myPoints.sum(1)  # 计算每个点的 x + y 和
    myPointsNew[0] = myPoints[np.argmin(add)]  # 左上角点具有最小的 x+y 值
    myPointsNew[3] = myPoints[np.argmax(add)]  # 右下角点具有最大的 x+y 值
    diff = np.diff(myPoints, axis=1)  # 计算每个点的 x - y 差值
    myPointsNew[1] = myPoints[np.argmin(diff)]  # 右上角点具有最小的 x-y 值
    myPointsNew[2] = myPoints[np.argmax(diff)]  # 左下角点具有最大的 x-y 值
    return myPointsNew


def biggestContour(contours):
    """
    从轮廓列表中找到面积最大的四边形轮廓。
    参数:
        contours: 轮廓列表。
    返回:
        biggest: 最大轮廓的坐标数组。
        max_area: 最大轮廓的面积。
    """
    biggest = np.array([])  # 初始化最大轮廓
    max_area = 0  # 初始化最大面积
    for i in contours:
        area = cv2.contourArea(i)  # 计算轮廓的面积
        if area > 5000:  # 忽略面积小于 5000 的轮廓
            peri = cv2.arcLength(i, True)  # 计算轮廓的周长
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 获取近似轮廓的顶点
            if area > max_area and len(approx) == 4:  # 如果轮廓面积最大且为四边形
                biggest = approx  # 更新最大轮廓
                max_area = area  # 更新最大面积
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    """
    根据四个顶点在图像上绘制矩形轮廓。
    参数:
        img: 要绘制矩形的图像。
        biggest: 包含矩形四个顶点的坐标数组。
        thickness: 矩形边框的粗细。
    返回:
        img: 绘制了矩形的图像。
    """
    # 使用线段将四个顶点连接，绘制出矩形
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img


def nothing(x):
    """
    空函数，用于滑动条回调，避免执行任何操作。
    """
    pass


def initializeTrackbars(initialTrackbarVals=0):
    """
    初始化用于 Canny 边缘检测的滑动条，以动态调整图像阈值。
    参数:
        initialTrackbarVals: 滑动条的初始值（默认为 0）。
    """
    cv2.namedWindow("Trackbars")  # 创建滑动条窗口
    cv2.resizeWindow("Trackbars", 360, 240)  # 调整窗口大小
    # 创建用于 Canny 边缘检测的两个滑动条
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    """
    获取滑动条的当前值，用于设置 Canny 边缘检测的上下阈值。
    返回:
        src: 当前的阈值（Threshold1, Threshold2）。
    """
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")  # 获取 Threshold1 滑动条的值
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")  # 获取 Threshold2 滑动条的值
    src = Threshold1, Threshold2  # 将阈值打包为元组
    return src
