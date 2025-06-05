import cv2
import numpy as np

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

    def create_mask(self, image_shape):
        """创建掩码"""
        return np.zeros(image_shape[:2], dtype=np.uint8)

    def update_display(self):
        """更新显示"""
        if self.original_image is not None:
            if self.drawing and self.current_mask is not None:
                # 绘制时显示红色标记
                temp = self.original_image.copy()
                temp[self.current_mask > 0] = [0, 0, 255]  # 用红色标记当前绘制区域
                if self.final_mask is not None:
                    # 显示已确认的修复效果
                    temp[self.final_mask > 0] = cv2.inpaint(self.original_image, self.final_mask, 3, cv2.INPAINT_TELEA)[self.final_mask > 0]
            else:
                # 非绘制状态显示修复效果
                temp = self.original_image.copy()
                if self.final_mask is not None:
                    temp[self.final_mask > 0] = cv2.inpaint(self.original_image, self.final_mask, 3, cv2.INPAINT_TELEA)[self.final_mask > 0]
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
        """鼠标回调函数，用于绘制掩码"""
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
            # 交互式绘制掩码
            self.original_image = image.copy()
            self.final_mask = None
            self.current_mask = None
            
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            print("使用鼠标左键绘制要修复的区域，按'r'键完成修复，按'q'键退出")
            print("按'+'键增加画笔大小，按'-'键减小画笔大小")
            print("按'c'键清除所有绘制区域")
            
            while True:
                self.update_display()
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):  # 按'r'键完成修复
                    break
                elif key == ord('q'):  # 按'q'键退出
                    cv2.destroyAllWindows()
                    return None
                elif key == ord('+'):  # 增加画笔大小
                    self.brush_size = min(50, self.brush_size + 2)
                    print(f"画笔大小: {self.brush_size}")
                elif key == ord('-'):  # 减小画笔大小
                    self.brush_size = max(2, self.brush_size - 2)
                    print(f"画笔大小: {self.brush_size}")
                elif key == ord('c'):  # 清除所有绘制区域
                    self.final_mask = None
                    self.current_mask = None
                    print("已清除所有绘制区域")
                    self.update_display()
                
            mask = self.final_mask
            cv2.destroyAllWindows()

        # 使用INPAINT_TELEA算法进行修复
        repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return repaired

    def add_text(self, image, text, position, font_scale=1.0, color=(0, 0, 0), thickness=2):
        """
        在图像上添加文字
        :param image: 输入图像
        :param text: 要添加的文字
        :param position: 文字位置 (x, y)
        :param font_scale: 字体大小
        :param color: 文字颜色
        :param thickness: 文字粗细
        :return: 添加文字后的图像
        """
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, text, position, font, font_scale, color, thickness)
        return result

    def repair_and_complete(self, image_path, text=None, text_position=None):
        """
        修复图像并补全文字
        :param image_path: 图像路径
        :param text: 要补全的文字
        :param text_position: 文字位置
        :return: 修复并补全后的图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None

        # 修复图像
        repaired = self.repair_image(image)
        if repaired is None:
            return None

        # 如果提供了文字和位置，则添加文字
        if text is not None and text_position is not None:
            repaired = self.add_text(repaired, text, text_position)

        return repaired