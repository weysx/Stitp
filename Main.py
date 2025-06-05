import cv2
import numpy as np
from ImagePreprocessing import preprocess
from TextRecognition import text_recognition


def main():
    # 读取图像
    ima_path = '/Users/weishixiong/Downloads/StitpProject/example4.png'
    img = cv2.imread(ima_path)

    if img is None:
        print("Error: Could not read image from path")
        exit()
    img_copy = img.copy()

    # 图像预处理
    # preprocess(img)  # 显示处理全过程
    preprocessed_img = preprocess(img)
    cv2.imshow('img_final', preprocessed_img)
    cv2.waitKey(0)

    # 将预处理好的图像进行文字识别
    text_recognition(preprocessed_img)

if __name__ == '__main__':
    main()
