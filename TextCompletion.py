import cv2
import numpy as np
from TextRecognition import text_recognition

class TextCompletion:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_thickness = 2
        self.font_color = (0, 0, 0)  # 黑色

    def analyze_text_style(self, image, text_region, text):
        """
        分析文字区域的样式
        :param image: 输入图像
        :param text_region: 文字区域坐标 (x, y, w, h)
        :param text: 要补全的文字
        :return: 字体大小、颜色等样式信息
        """
        x, y, w, h = text_region
        
        # 根据区域大小估算字体大小
        self.font_scale = min(w / (len(text) * 20), h / 30)  # 20和30是经验值
        
        return {
            'color': (0, 0, 0),  # 固定为黑色
            'scale': self.font_scale,
            'thickness': self.font_thickness
        }

    def match_font_style(self, image, text_region, text):
        """
        匹配文字样式
        :param image: 输入图像
        :param text_region: 文字区域坐标
        :param text: 要补全的文字
        :return: 样式信息
        """
        return self.analyze_text_style(image, text_region, text)

    def add_text(self, image, text, region, style=None):
        """
        在图像的指定区域内居中添加文字
        :param image: 输入图像
        :param text: 要添加的文字
        :param region: 文字区域 (x, y, w, h)
        :param style: 文字样式
        :return: 添加文字后的图像
        """
        if style:
            self.font_scale = style.get('scale', self.font_scale)
            self.font_thickness = style.get('thickness', self.font_thickness)

        x, y, w, h = region
        # 获取文字尺寸
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        # 计算居中位置
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2  # 注意OpenCV的y是基线

        result = image.copy()
        cv2.putText(result, text, (text_x, text_y), self.font, 
                   self.font_scale, (0, 0, 0), self.font_thickness)
        return result

    def complete_text(self, image, text_region, text):
        """
        补全文字
        :param image: 输入图像
        :param text_region: 文字区域坐标 (x, y, w, h)
        :param text: 要补全的文字
        :return: 补全后的图像
        """
        # 匹配文字样式
        style = self.match_font_style(image, text_region, text)
        
        # 居中添加文字
        return self.add_text(image, text, text_region, style)

def main():
    # 测试代码
    completion = TextCompletion()
    
    # 读取图像
    image = cv2.imread('example1.png')
    if image is None:
        print("无法读取图像")
        return

    # 识别文字
    result = text_recognition(image)
    if result:
        # 获取第一个识别到的文字区域
        words_result = result.get('words_result', [])
        if words_result:
            # 这里需要根据实际情况调整文字区域坐标
            text_region = (100, 100, 200, 50)  # 示例坐标
            text = "补全的文字"
            
            # 补全文字
            completed = completion.complete_text(image, text_region, text)
            
            # 显示结果
            cv2.imshow('Completed Image', completed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 