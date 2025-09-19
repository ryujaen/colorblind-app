# test_cvd.py
import json
import re
from pathlib import Path
import streamlit as st

# ---------- 입력 매핑 유틸 ----------
def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"\s+", "", s)

def _digits(s: str) -> str | None:
    m = re.findall(r"\d+", s or "")
    return "".join(m) if m else None

def map_input_to_choice(choices: list[str], user_text: str) -> str | None:
    """자유 입력(숫자/텍스트)을 기존 choice 문자열로 매핑"""
    if not user_text:
        return None

    t = _canon(user_text)

    # 1) 숫자 매칭 (예: "12", "12번", "12." 등)
    d = _digits(t)
    if d:
        for c in choices:
            cd = _digits(c)
            if cd and cd == d:
                return c

    # 2) '안 보임' 계열
    if any(kw in t for kw in ["안보임", "안보여", "없음", "없다", "none", "no", "보이지않음", "x"]):
        for c in choices:
            if _canon(c).startswith("안보임") or "not" in _canon(c):
                return c

    # 3) '다르게 보임' 계열
    if any(kw in t for kw in ["다르게", "모름", "?", "other", "unknown"]):
        for c in choices:
            if "다르게" in c or "other" in _canon(c):
                return c

    # 4) 완전 일치
    for c in choices:
        if _canon(c) == t:
            return c

    return None

# ---------- 데이터/규칙 ----------
PLATES_PATH = Path("data/plates.json")

@st.cache_data
def load_plates():
    return json.loads(PLATES_PATH.read_text(encoding="utf-8"))

def _acc(votes, delta):
    for k, v in delta.items():
        votes[k] = votes.get(k, 0) + v

def _infer(votes):
    ordered = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    top, second = ordered[0], ordered[1]
    ctype = top[0]  # 'normal' | 'protan' | 'deutan' | 'tritan'
    gap = top[1] - second[1]

    # 심도 간단 규칙 (갭 기반)
    if ctype == "normal":
        severity = 0
    else:
        if   gap >= 4: severity = 85
        elif gap >= 3: severity = 65
        elif gap >= 2: severity = 45
        else: severity = 25

    # 앱 내부 키로 변환
    cvd_key = {
        "protan":  "protanomaly",
        "deutan":  "deuteranomaly",
        "tritan":  "tritanomaly",
        "normal":  "normal",
    }[ctype]
    return cvd_key, severity, ordered

def _order_adaptive(plates, votes):
    base_ids = {"P01", "P02", "P12"}  # 공통 3문항
    base = [p for p in plates if p["id"] in base_ids]
    rest = [p for p in plates if p["id"] not in base_ids]
    if not votes or max(votes, key=votes.get) == "normal":
        return base + rest
    top = max(votes, key=votes.get)

    # 가중치에 top이 언급되는 문항 우선
    def targets(p):
        for w in p["weights"].values():
            if top in w:
                return True
        return False

    pri = [p for p in rest if targets(p)]
    oth = [p for p in rest if p not in pri]
    return base + pri + oth

# ---------- 메인: 자유 입력형 검사 ----------
def run_color_vision_test():
    plates = load_plates()
    st.subheader("👁️ 색각 간이 검사 (6~8문항)")
    st.caption("밝은 화면에서 50~70cm 거리 권장")

    # 세션 초기값
    st.session_state.setdefault("tc_votes", {"normal": 0, "protan": 0, "deutan": 0, "tritan": 0})
    st.session_state.setdefault("tc_run", 0)  # 위젯 키 변경용 시퀀스

    # 초기화 버튼
    if st.button("⬅️ 처음부터 다시"):
        for k in list(st.session_state.keys()):
            if k.startswith("tc_free_"):
                del st.session_state[k]
        st.session_state["tc_votes"] = {"normal": 0, "protan": 0, "deutan": 0, "tritan": 0}
        st.session_state["tc_run"] += 1
        st.experimental_rerun()

    order = _order_adaptive(plates, st.session_state["tc_votes"])

    asked = 0
    for p in order:
        if asked >= 8:
            break

        st.image(p["img"], use_container_width=True)

        # 텍스트 직접 입력
        user_ans = st.text_input(
            label=p["question"],
            placeholder="예: 12  /  안 보임  /  다르게 보임",
            key=f"tc_free_{st.session_state['tc_run']}_{p['id']}",
        )

        choice = map_input_to_choice(p["choices"], user_ans)

        if user_ans:  # 뭔가 입력했을 때만 처리
            if choice is None:
                st.caption("⚠️ 인식되지 않은 입력이에요. 예: 12 / 안 보임 / 다르게 보임")
            else:
                _acc(st.session_state["tc_votes"], p["weights"].get(choice, {}))
                asked += 1

        st.divider()

    # 결과 버튼
    if st.button("결과 보기", key="tc_result_btn"):
        cvd_key, severity, ordered = _infer(st.session_state["tc_votes"])
        if cvd_key == "normal":
            st.success("정상 시각으로 추정됩니다.")
        else:
            st.success(f"예상 유형: **{cvd_key}**, 심도: **{severity}**")
        return cvd_key, severity

    # 항상 튜플 반환(호출부 언패킹 에러 방지)
    return (None, None)
