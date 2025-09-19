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

    # 초기 세션 키 준비
    st.session_state.setdefault("tc_votes", {"normal":0,"protan":0,"deutan":0,"tritan":0})
    st.session_state.setdefault("tc_run", 0)  # 위젯 키에 섞을 런 번호

    # ▶️ 초기화 버튼
    if st.button("⬅️ 처음부터 다시"):
        # 기존 답변 위젯 상태/투표 모두 지우기
        for k in list(st.session_state.keys()):
            if k.startswith("tc_ans_"):
                del st.session_state[k]
        st.session_state["tc_votes"] = {"normal":0,"protan":0,"deutan":0,"tritan":0}
        st.session_state["tc_run"] += 1
        st.experimental_rerun()

    order = _order_adaptive(plates, st.session_state["tc_votes"])

    asked = 0
    for p in order:
        if asked >= 8:
            break
        st.image(p["img"], use_container_width=True)

        # index=None: 기본 미선택
        try:
            choice = st.radio(
                p["question"],
                p["choices"],           # '선택 안 함' 제거
                index=None,             # ✅ 미선택 시작
                key=f"tc_ans_{st.session_state['tc_run']}_{p['id']}"
            )
        except TypeError:
            # 구버전 Streamlit 호환(라디오는 None 미지원일 때)
            choice = st.selectbox(
                p["question"],
                p["choices"],
                index=None,
                placeholder="선택해 주세요",
                key=f"tc_ans_{st.session_state['tc_run']}_{p['id']}"
            )

        if choice is not None:
            _acc(st.session_state["tc_votes"], p["weights"].get(choice, {}))
            asked += 1

        st.divider()
