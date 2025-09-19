# test_cvd.py
import json
from pathlib import Path
import streamlit as st

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
    ctype = top[0] # 'normal' | 'protan' | 'deutan' | 'tritan'
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
    cvd_key = {"protan":"protanomaly", "deutan":"deuteranomaly", "tritan":"tritanomaly", "normal":"normal"}[ctype]
    return cvd_key, severity, ordered

def _order_adaptive(plates, votes):
    base_ids = {"P01","P02","P12"}              # 공통 3문항
    base = [p for p in plates if p["id"] in base_ids]
    rest = [p for p in plates if p["id"] not in base_ids]
    if not votes or max(votes, key=votes.get) == "normal":
        return base + rest
    top = max(votes, key=votes.get)
    # 가중치에 top이 언급되는 문항 우선
    def targets(p):
        for w in p["weights"].values():
            if top in w: return True
        return False
    pri = [p for p in rest if targets(p)]
    oth = [p for p in rest if p not in pri]
    return base + pri + oth

def run_color_vision_test():
    plates = load_plates()
    st.subheader("👁️ 색각 간이 검사 (6~8문항)")
    st.caption("밝은 화면에서 50~70cm 거리 권장")

    st.session_state.setdefault("tc_votes", {"normal":0,"protan":0,"deutan":0,"tritan":0})

    order = _order_adaptive(plates, st.session_state["tc_votes"])

    asked = 0
    for p in order:
        if asked >= 8:
            break
        st.image(p["img"], use_container_width=True)

        # 기본값 없애기 위해 "선택 안 함" 추가
        choices = p["choices"]
        choice = st.radio(
            p["question"],
            choices,
            index=None,
            key=f"tc_ans_{p['id']}")

        if choice is not None:
            _acc(st.session_state["tc_votes"], p["weights"].get(choice, {}))
            asked += 1

        st.divider()

    if st.button("결과 보기", key="tc_result_btn"):
        cvd_key, severity, ordered = _infer(st.session_state["tc_votes"])
        if cvd_key == "normal":
            st.success("정상 시각으로 추정됩니다.")
        else:
            st.success(f"예상 유형: **{cvd_key}**, 심도: **{severity}**")
        return cvd_key, severity
    return None, None