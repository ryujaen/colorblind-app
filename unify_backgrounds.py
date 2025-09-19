# unify_backgrounds.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img

def replace_bg(img: Image.Image, bg_color=(0, 0, 0), tol=30) -> Image.Image:
    """
    - 모서리(네 귀퉁이) 픽셀 평균색을 배경 후보로 보고
    - 그 색과의 거리(tol) 내에 있는 픽셀을 모두 bg_color로 바꿉니다.
    - 알파가 있는 PNG는 원본 알파 유지
    """
    arr = np.array(img)  # H x W x 4 (RGBA)
    h, w, _ = arr.shape

    # 코너 샘플들: (0,0), (0,w-1), (h-1,0), (h-1,w-1)
    corners = np.array([
        arr[2:10, 2:10, :3].reshape(-1, 3),
        arr[2:10, w-10:w-2, :3].reshape(-1, 3),
        arr[h-10:h-2, 2:10, :3].reshape(-1, 3),
        arr[h-10:h-2, w-10:w-2, :3].reshape(-1, 3),
    ])
    corner_mean = corners.reshape(-1, 3).mean(axis=0)

    rgb = arr[:, :, :3].astype(np.int16)
    dist = np.linalg.norm(rgb - corner_mean, axis=2)  # 유클리드 거리
    mask = dist <= tol

    out = arr.copy()
    out[:, :, :3][mask] = np.array(bg_color, dtype=np.uint8)

    return Image.fromarray(out, mode="RGBA")

def pad_on_canvas(img: Image.Image, bg_color=(0, 0, 0), margin=0) -> Image.Image:
    """
    - 원본 이미지를 그대로 **캔버스 위에 올려** 배경을 강제합니다.
    - margin 픽셀만큼 여백을 두고 가운데 배치.
    """
    w, h = img.size
    canvas = Image.new("RGBA", (w + margin*2, h + margin*2),
                       color=(*bg_color, 255))
    canvas.paste(img, (margin, margin), mask=img if img.mode == "RGBA" else None)
    return canvas

def process_dir(src: Path, dst: Path, bg: str, mode: str, tol: int, margin: int):
    dst.mkdir(parents=True, exist_ok=True)
    bg_color = (0, 0, 0) if bg == "black" else (255, 255, 255)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"[warn] No images found under {src}")
        return

    print(f"[info] {len(files)} images found. mode={mode}, bg={bg}, tol={tol}, margin={margin}")
    for p in files:
        img = load_image(p)
        if mode == "replace":
            out = replace_bg(img, bg_color=bg_color, tol=tol)
        else:
            out = pad_on_canvas(img, bg_color=bg_color, margin=margin)

        # 원본 폴더 구조 유지
        rel = p.relative_to(src)
        save_path = (dst / rel).with_suffix(".png")  # 통일 저장
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(save_path)
        print(f"[ok] {rel} -> {save_path.relative_to(dst)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unify background color of Ishihara plates.")
    ap.add_argument("--src", required=True, help="원본 이미지 폴더")
    ap.add_argument("--dst", required=True, help="결과 저장 폴더")
    ap.add_argument("--bg", choices=["black", "white"], default="black", help="배경색")
    ap.add_argument("--mode", choices=["replace", "pad"], default="replace",
                    help="replace: 배경색 유사 픽셀을 치환 / pad: 캔버스에 올려 강제 배경")
    ap.add_argument("--tol", type=int, default=30, help="replace 모드에서 허용 색 거리(작을수록 엄격)")
    ap.add_argument("--margin", type=int, default=0, help="pad 모드에서 여백(px)")
    args = ap.parse_args()

    process_dir(Path(args.src), Path(args.dst), args.bg, args.mode, args.tol, args.margin)
