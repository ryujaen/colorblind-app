import numpy as np
import cv2

def apply_circle_mask(image):
    image = np.array(image) 
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    radius = min(center) - 20  # 약간 여백 줌

    # 원형 마스크 만들기
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # 원형 부분은 이미지 유지, 바깥은 흰색 대신 회색
    result = np.ones_like(image, dtype=np.uint8)
    for i in range(3):  # RGB 각 채널별 적용
        result[:, :, i] = np.where(mask == 255, image[:, :, i], 200)  # 바깥을 회색(200)으로 설정

    return result
