import cv2
import numpy as np

# -------- 공통 유틸 --------
def _clip_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def _apply_clahe_l_channel(lab, clip_limit=2.0, tile_grid_size=(8, 8)):
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L = clahe.apply(L)
    return cv2.merge([L, A, B])

def _boost_saturation_hsv(rgb, sat_scale=1.10):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    S = np.clip(S.astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    hsv = cv2.merge([H, S, V])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def _shift_hue_for_reds(rgb, deg=6):
    """
    빨강 계열만 살짝 푸른쪽으로 회전 (프로타노피아 분리 강화).
    deg: 0~179 (OpenCV HSV 기준 한 바퀴 180)
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    # 빨간색 영역: [0~15] or [165~179]
    mask_r1 = (H <= 15)
    mask_r2 = (H >= 165)
    H = H.astype(np.int16)
    H[mask_r1] = (H[mask_r1] + deg) % 180
    H[mask_r2] = (H[mask_r2] - deg) % 180
    H = H.astype(np.uint8)
    hsv = cv2.merge([H, S, V])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# -------- 유형별 보정 --------
def _correct_protanopia(rgb):
    """
    프로타노피아(적원추 결손):
    - Lab a축(녹-적) 대비 강화(+bias)로 적/녹 분리
    - 빨강계열 hue를 살짝 푸른쪽으로 이동
    - 전체 채도 소폭 증가, L 대비(명암) 보정
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    A = A.astype(np.float32)
    # a 축을 살짝 양의 방향으로 밀어 적-녹 분리 강화
    A = (A - 128.0) * 1.15 + 128.0 + 8.0
    A = np.clip(A, 0, 255).astype(np.uint8)

    lab = cv2.merge([L, A, B])
    lab = _apply_clahe_l_channel(lab, clip_limit=2.0)
    rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb2 = _shift_hue_for_reds(rgb2, deg=6)    # 빨강만 국소 hue 이동
    rgb2 = _boost_saturation_hsv(rgb2, sat_scale=1.10)
    return rgb2

def _correct_deuteranopia(rgb):
    """
    듀테라노피아(중원추 결손):
    - Lab a축을 음의 방향으로 이동(초록쪽으로), cyan/teal쪽 분리 강화
    - 전체 채도 소폭 증가, L 대비 보정
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    A = A.astype(np.float32)
    # a 축을 약간 음의 방향으로 이동 → 초록/청록 분리 강화
    A = (A - 128.0) * 1.10 + 128.0 - 12.0
    A = np.clip(A, 0, 255).astype(np.uint8)

    lab = cv2.merge([L, A, B])
    lab = _apply_clahe_l_channel(lab, clip_limit=2.0)
    rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb2 = _boost_saturation_hsv(rgb2, sat_scale=1.10)
    return rgb2

def _correct_tritanopia(rgb):
    """
    트리타노피아(청원추 결손):
    - Lab b축(청-황) 대비 강화 및 약간 음/양 이동
    - 노랑/파랑 분리, 채도 증가, L 대비 보정
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    B = B.astype(np.float32)
    # b 축을 살짝 스케일 + 오프셋: 파랑/노랑 구분 강조
    B = (B - 128.0) * 1.15 + 128.0 - 10.0
    B = np.clip(B, 0, 255).astype(np.uint8)

    lab = cv2.merge([L, A, B])
    lab = _apply_clahe_l_channel(lab, clip_limit=2.2)
    rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb2 = _boost_saturation_hsv(rgb2, sat_scale=1.12)
    return rgb2

# -------- 공개 API --------
def apply_colorblind_filter(img_bgr, color_type):
    """
    입력: BGR(OpenCV) 이미지, color_type 문자열(한/영 혼용 지원)
    출력: BGR(OpenCV) 보정 이미지
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 타입 정규화
    ct = (color_type or "").strip().lower()
    if "protan" in ct or "적색" in ct:
        out_rgb = _correct_protanopia(rgb)
    elif "deuter" in ct or "녹색" in ct:
        out_rgb = _correct_deuteranopia(rgb)
    elif "tritan" in ct or "청색" in ct:
        out_rgb = _correct_tritanopia(rgb)
    else:
        # 모르면 원본 반환
        out_rgb = rgb

    # RGB -> BGR
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return _clip_uint8(out_bgr)
