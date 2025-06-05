import cv2
import numpy as np
from TextCompletion import TextCompletion

class ImageRepair:
    def __init__(self):
        self.window_name = "Image Repair Tool"
        self.drawing = False
        self.mask = None
        self.original_image = None
        self.current_image = None
        self.last_x, self.last_y = -1, -1
        self.brush_size = 20
        self.repaired_image = None
        self.min_distance = 2  # 最小绘制距离
        self.current_mask = None  # 当前正在绘制的掩码
        self.final_mask = None  # 最终确认的掩码
        self.text_completion = TextCompletion()
        self.text_regions = []  # 存储文字区域信息
        self.adjusting_text = False  # 是否正在调整文字区域
        self.current_text_region = None  # 当前正在调整的文字区域
        self.text_region_start = None  # 文字区域起始点

    def create_mask(self, image_shape):
        """创建掩码"""
        return np.zeros(image_shape[:2], dtype=np.uint8)

    def update_display(self):
        """更新显示"""
        if self.original_image is not None:
            temp = self.original_image.copy()
            
            # 显示修复效果
            if self.final_mask is not None:
                # 先进行图像修复
                repaired = cv2.inpaint(self.original_image, self.final_mask, 3, cv2.INPAINT_TELEA)
                temp = repaired.copy()
            
            # 显示当前绘制区域
            if self.drawing and self.current_mask is not None:
                temp[self.current_mask > 0] = [0, 0, 255]  # 红色
            
            # 显示文字区域
            for region in self.text_regions:
                x, y, w, h = region
                cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框
            
            # 显示正在调整的文字区域
            if self.adjusting_text and self.text_region_start is not None:
                x, y = self.text_region_start
                cv2.rectangle(temp, (x, y), (self.last_x, self.last_y), (255, 0, 0), 2)  # 蓝色框
            
            cv2.imshow(self.window_name, temp)

    def draw_line(self, x1, y1, x2, y2):
        """绘制线条，添加距离检查"""
        # 计算两点之间的距离
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance < self.min_distance:
            return
        
        # 使用抗锯齿的线条绘制
        cv2.line(self.current_mask, (x1, y1), (x2, y2), 255, self.brush_size, cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，用于绘制掩码或文本区域"""
        if self.adjusting_text:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.text_region_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.text_region_start is not None:
                self.last_x, self.last_y = x, y
                self.update_display()
            elif event == cv2.EVENT_LBUTTONUP:
                if self.text_region_start is not None:
                    x1, y1 = self.text_region_start
                    x2, y2 = x, y
                    # 确保坐标是有序的
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    self.text_regions.append((x, y, w, h))
                    self.text_region_start = None
                    self.update_display()
                    self.adjusting_text = False  # 新增，松开鼠标后退出调整模式
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_x, self.last_y = x, y
            # 创建新的当前掩码
            self.current_mask = self.create_mask(self.original_image.shape)
            # 在点击位置画一个点
            cv2.circle(self.current_mask, (x, y), self.brush_size//2, 255, -1, cv2.LINE_AA)
            self.update_display()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 绘制线条
                self.draw_line(self.last_x, self.last_y, x, y)
                self.last_x, self.last_y = x, y
                self.update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # 在松开鼠标时画最后一个点
                cv2.circle(self.current_mask, (x, y), self.brush_size//2, 255, -1, cv2.LINE_AA)
                # 将当前掩码合并到最终掩码
                if self.final_mask is None:
                    self.final_mask = self.current_mask.copy()
                else:
                    self.final_mask = cv2.bitwise_or(self.final_mask, self.current_mask)
                self.current_mask = None
                self.update_display()

    def repair_image(self, image, mask=None):
        """
        修复图像中的损坏部分
        :param image: 输入图像
        :param mask: 损坏区域的掩码，如果为None则使用交互式绘制
        :return: 修复后的图像
        """
        if mask is None:
            self.original_image = image.copy()
            self.final_mask = None
            self.current_mask = None
            self.text_regions = []
            repaired = None
            repaired_done = False
            editing_text_regions = False

            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            print("使用鼠标左键绘制要修复的区域，按'r'键完成修复，按'q'键退出")
            print("按'+'键增加画笔大小，按'-'键减小画笔大小")
            print("按'c'键清除所有绘制区域")
            print("按'a'键添加/调整文本区域")
            print("按't'键补全文字")

            while True:
                if repaired_done:
                    temp = repaired.copy()
                    # 显示所有文本区域
                    for region in self.text_regions:
                        x, y, w, h = region
                        cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 实时显示正在调整的文本区域
                    if self.adjusting_text and self.text_region_start is not None:
                        x, y = self.text_region_start
                        cv2.rectangle(temp, (x, y), (self.last_x, self.last_y), (255, 0, 0), 2)
                    cv2.imshow(self.window_name, temp)
                else:
                    self.update_display()

                key = cv2.waitKey(1) & 0xFF

                if not repaired_done:
                    if key == ord('r'):
                        repaired = cv2.inpaint(self.original_image, self.final_mask, 3, cv2.INPAINT_TELEA)
                        repaired_done = True
                        print("修复完成！可继续按'a'添加/调整文本区域，按't'补全文字，按'q'退出")
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        return None
                    elif key == ord('+'):
                        self.brush_size = min(50, self.brush_size + 2)
                        print(f"画笔大小: {self.brush_size}")
                    elif key == ord('-'):
                        self.brush_size = max(2, self.brush_size - 2)
                        print(f"画笔大小: {self.brush_size}")
                    elif key == ord('c'):
                        self.final_mask = None
                        self.current_mask = None
                        self.text_regions = []
                        print("已清除所有绘制区域")
                        self.update_display()
                else:
                    if key == ord('a'):
                        print("请用鼠标左键拖动选择文本区域，松开后区域会高亮显示。可多次添加。")
                        editing_text_regions = True
                        self.adjusting_text = True  # 新增，允许实时显示边框
                        cv2.setMouseCallback(self.window_name, self.mouse_callback)
                    elif key == ord('t'):
                        if self.text_regions:
                            text = input("请输入要补全的文字：")
                            repaired = self.complete_text(repaired, text)
                            print("已补全文字，可继续添加文本区域或补全文字，按q退出")
                        else:
                            print("未检测到文本区域，请先按a添加")
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        return repaired

        # 使用INPAINT_TELEA算法进行修复
        repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return repaired

    def complete_text(self, image, text):
        """
        补全文字
        :param image: 输入图像
        :param text: 要补全的文字
        :return: 补全后的图像
        """
        if not self.text_regions:
            return image

        result = image.copy()
        for region in self.text_regions:
            result = self.text_completion.complete_text(result, region, text)

        return result

def main():
    # 测试代码
    repair = ImageRepair()
    
    # 读取图像
    image = cv2.imread('example1.png')
    if image is None:
        print("无法读取图像")
        return

    # 修复图像
    repaired = repair.repair_image(image)
    if repaired is not None:
        cv2.imshow('Repaired Image', repaired)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()