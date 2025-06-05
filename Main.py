import cv2
import numpy as np
from ImagePreprocessing import preprocess
from TextRecognition import text_recognition
from ImageRepair import ImageRepair


def main():
    # 读取图像
    ima_path = '/Users/weishixiong/Downloads/StitpProject/example2.jpg'
    img = cv2.imread(ima_path)

    if img is None:
        print("Error: Could not read image from path")
        exit()
    img_copy = img.copy()

    # 创建图像修复工具实例
    repair_tool = ImageRepair()
    
    # 修复图像
    print("开始修复图像...")
    repaired_img = repair_tool.repair_image(img)
    
    if repaired_img is None:
        print("图像修复已取消")
        return
    
    # 显示修复后的图像
    cv2.imshow('Repaired Image', repaired_img)
    cv2.waitKey(0)
    
    # 图像预处理
    print("开始图像预处理...")
    preprocessed_img = preprocess(repaired_img)
    cv2.imshow('Preprocessed Image', preprocessed_img)
    cv2.waitKey(0)

    # 将预处理好的图像进行文字识别
    print("开始文字识别...")
    text_recognition(preprocessed_img)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
