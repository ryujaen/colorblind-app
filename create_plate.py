from PIL import Image,ImageDraw
from image_utils import apply_circle_mask
import random

# 설정값
img_size = 400
dot_radius = 7
num_dots = 1300

# 색상: Protanopia(적색맹)용 예시
bg_color = (100, 180, 100)      # 배경 (녹색 계열)
num_color = (180, 100, 100)     # 숫자 (적색 계열)

# 이미지 생성
image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
draw = ImageDraw.Draw(image)

# 숫자 모양 정의 (예: 숫자 5)
def in_number(x, y):
    cx, cy = img_size // 2, img_size // 2
    dx, dy = x - cx, y - cy
    r = (dx**2 + dy**2)**0.5

    if r > img_size // 2 - 10:
        return False

    # 숫자 5 모양 정의
    if 100 < x < 200 and 100 < y < 130:
        return True
    if 100 < x < 120 and 100 < y < 200:
        return True
    if 100 < x < 200 and 180 < y < 200:
        return True
    if 180 < x < 200 and 180 < y < 260:
        return True
    if 100 < x < 200 and 240 < y < 260:
        return True

    return False

# 도트 뿌리기
for _ in range(num_dots):
    x = random.randint(dot_radius, img_size - dot_radius)
    y = random.randint(dot_radius, img_size - dot_radius)
    color = num_color if in_number(x, y) else bg_color
    draw.ellipse(
        (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
        fill=color
    )

# 이미지 저장 및 보기
image.save("plate_custom5.png")
image.show()


# 이미지 저장
image= apply_circle_mask(image)  # 원형 마스크 적용
Image.fromarray(image).save("plate_custom5.png")
