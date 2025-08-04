import numpy as np
import cv2

# 색각이상 유형별 변환 행렬
colorblind_matrices = {
    'Protanopia (적색맹)': np.array([[0.567, 0.433, 0],
                                     [0.558, 0.442, 0],
                                     [0,     0.242, 0.758]]),
    'Deuteranopia (녹색맹)': np.array([[0.625, 0.375, 0],
                                      [0.7,   0.3,   0],
                                      [0,     0.3,   0.7]]),
    'Tritanopia (청색맹)': np.array([[0.95,  0.05,   0],
                                    [0,     0.433, 0.567],
                                    [0,     0.475, 0.525]])
}

# 필터 적용 함수
def apply_colorblind_filter(image, filter_name):
    if filter_name not in colorblind_matrices:
        return image
    matrix = colorblind_matrices[filter_name]
    return cv2.transform(image, matrix)