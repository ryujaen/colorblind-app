# test_cvd.py
from __future__ import annotations

import json
import re
from pathlib import Path
import streamlit as st
from PIL import Image
from image_utils import make_circular_rgba

# ==============================
# 입력 매핑 유틸
# ==============================
def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", "", s)

def _digits(s: str) -> str | None:
    m = re.findall(r"\d+", s or "")
    return "".join(m) if m else None

def map_input_to_choice(choices: list[str], user_text: str) -> str | None:
    """
    자유 입력(숫자/텍스트)을 기존 choice 문자열로 매핑
    - '12' → '12'
    - '안보임', 'no', 'x' → '안보임' 계열
    - '다르게 보임', 'other', '?' → '다르게 보임' 계열
    """
    if not user_text:
        return None

    t = _canon(user_text)

    # 1) 숫자 매칭
    d = _digits(t)
    if d:
        for c in choices:
            cd = _digits(c)
            if cd and cd == d:
                return c

    # 2) '안 보임' 계열
    if any(kw in t for kw in ["안보임", "안보여", "없음", "없다", "none", "no", "보이지않음", "x"]):
        for c in choices:
            cc = _canon(c)
            if cc.startswith("안보임") or "not" in cc:
                return c

    # 3) '다르게 보임' 계열
    if any(kw in t for kw in ["다르게", "모름", "other", "unknown", "?"]):
        for c in choices:
            cc = _canon(c)
            if "다르게보임" in cc or "other" in cc:
                return c

    # 4) 완전 일치
    for c in choices:
        if _canon(c) == t:
            return c

    return None


# ==============================
# 데이터 로딩 (BOM 안전)
# ==============================
PLATES_PATH = Path("data/plates.json")

@st.cache_data(show_spinner=False)
def load_plates() -> list[dict]:
    """
    plates.json을 BOM 유무에 상관없이 안전하게 로드.
    Streamlit 캐시로 재사용.
    """
    text = PLATES_PATH.read_text(encoding="utf-8-sig")  # <-- BOM 안전
    data = json.loads(text)
    # 예상 스키마 예시: [{"id":"P01","img":"data/plates/P01.png","question":"…","choices":["12","안 보임","다르게 보임"],"weights":{"12":{"normal":1},…}}]
    return data


# ==============================
# 스코어 누적/추론/문항 순서
# ==============================
def _acc(votes: dict[str, int], delta: dict[str, int]) -> None:
    for k, v in (delta or {}).items():
        votes[k] = votes.get(k, 0) + int(v)

def _infer(votes: dict[str, int]) -> tuple[str, int, list[tuple[str, int]]]:
    # 키 없을 때도 안전하게 0 기본값
    base = {k: int(votes.get(k, 0)) for k in ["normal", "protan", "deutan", "tritan"]}
    ordered = sorted(base.items(), key=lambda x: x[1], reverse=True)
    top, second = ordered[0], ordered[1]
    ctype = top[0]
    gap = top[1] - second[1]

    if ctype == "normal":
        severity = 0
    else:
        if   gap >= 4: severity = 85
        elif gap >= 3: severity = 65
        elif gap >= 2: severity = 45
        else: severity = 25

    cvd_key = {
        "protan":  "protanomaly",
        "deutan":  "deuteranomaly",
        "tritan":  "tritanomaly",
        "normal":  "normal",
    }[ctype]
    return cvd_key, severity, ordered

def _order_adaptive(plates: list[dict], votes: dict[str, int]) -> list[dict]:
    """
    초반에는 기본 세 문항으로 시작하고, 중간 결과에 따라 관련 문항을 우선 제시.
    """
    base_ids = {"P01", "P02", "P12"}
    base = [p for p in plates if p.get("id") in base_ids]
    rest = [p for p in plates if p.get("id") not in base_ids]

    if not votes or (max(votes, key=votes.get) if votes else "normal") == "normal":
        return base + rest

    top = max(votes, key=votes.get)

    def targets(p: dict) -> bool:
        w = (p.get("weights") or {}).values()
        for d in w:
            if top in (d or {}):
                return True
        return False

    pri = [p for p in rest if targets(p)]
    oth = [p for p in rest if p not in pri]
    return base + pri + oth


# ==============================
# 메인 엔트리 (앱에서 호출)
# ==============================
def run_color_vision_test() -> tuple[str | None, int | None]:
    plates = load_plates()

    st.subheader("👁️ 색각 간이 검사 (6~8문항)")
    st.caption("밝은 화면에서 50~70cm 거리 권장 · 본 결과는 의학적 진단이 아닙니다.")

    # 세션 상태 초기화
    st.session_state.setdefault("tc_votes", {"normal": 0, "protan": 0, "deutan": 0, "tritan": 0})
    st.session_state.setdefault("tc_run", 0)

    # 리셋 버튼
    if st.button("⬅️ 처음부터 다시"):
        # 이번 러닝에서 만든 입력 key만 제거
        for k in [k for k in st.session_state.keys() if k.startswith("tc_free_")]:
            del st.session_state[k]
        st.session_state["tc_votes"] = {"normal": 0, "protan": 0, "deutan": 0, "tritan": 0}
        st.session_state["tc_run"] += 1
        st.rerun()  # (experimental_rerun 대체)

    order = _order_adaptive(plates, st.session_state["tc_votes"])

    asked = 0
    for p in order:
        if asked >= 8:
            break

        # 이미지 표시 (원 밖 투명 처리)
        img_path = p.get("img")
        plate = Image.open(img_path).convert("RGBA")
        plate = make_circular_rgba(plate, margin=2)  # 필요하면 margin 조절
        st.image(plate, use_container_width=True)

        # 자유 입력
        user_ans = st.text_input(
            label=p.get("question", "보이는 숫자/패턴을 입력하세요."),
            placeholder="예: 12  /  안 보임  /  다르게 보임",
            key=f"tc_free_{st.session_state['tc_run']}_{p.get('id','')}",
        )

        # 선택 매핑
        choice = map_input_to_choice(list(p.get("choices", [])), user_ans)

        # 입력이 있으면 채점
        if user_ans:
            if choice is None:
                st.caption("⚠️ 인식되지 않은 입력이에요. 예: 12 / 안 보임 / 다르게 보임")
            else:
                weights = (p.get("weights") or {}).get(choice, {})
                _acc(st.session_state["tc_votes"], weights)
                asked += 1

        st.divider()

    # 결과 보기
    if st.button("결과 보기", key="tc_result_btn"):
        cvd_key, severity, ordered = _infer(st.session_state["tc_votes"])
        if cvd_key == "normal":
            st.success("정상 시각으로 추정됩니다.")
        else:
            st.success(f"예상 유형: **{cvd_key}**, 심도: **{severity}**")
        return cvd_key, severity

    return (None, None)
