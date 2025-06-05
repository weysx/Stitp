import requests
import base64
import cv2

def text_recognition(img):
    # 设置访问令牌 access_token，用于身份验证，确保可以访问百度的OCR API
    access_token = '24.877dff5c288a9a9cc3c7421dc4ffb6fa.2592000.1751539175.282335-115986366'
    # OCR API请求的URL，使用通用文字识别接口
    ocr_url = f'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={access_token}'

    # 将 OpenCV 读取的图像编码为 JPEG 格式并进行 base64 编码，以便于在 HTTP 请求中传输
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 设置请求头和数据
    # 创建请求数据，包含图像的 Base64 编码字符串
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'image': img_base64}

    # 使用 requests.post 方法发送 POST 请求到指定的 OCR API URL，附带请求头和数据
    response = requests.post(ocr_url, headers=headers, data=data)

    # 处理返回结果
    if response.status_code == 200:
        result = response.json() # 解析 JSON 格式的响应数据
        print("OCR 识别结果:")
        for item in result.get('words_result', []):
            print(item.get('words'))
        return result
    else:
        print("请求失败，状态码:", response.status_code)
        print("错误信息:", response.text)
        return None
