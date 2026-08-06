# -*- coding: utf-8 -*-
"""
DoctorOcr — OCR 결과 정성 분석 뷰어 (Streamlit)
================================================
result CSV (true/pred/match/label/path/group/cer) 를 읽어
성공 / 부분실패 / 실패 3-카테고리로 이미지 갤러리 검수를 지원한다.

실행:
    doctor_ocr_v3_1/venv/bin/python -m streamlit run utils/result_viewer.py

분류 기준 (cer 컬럼):
    성공       cer == 0
    부분실패   0 < cer <= 0.3   (일부 문자 인식, 예: 'Devixil'→'Denixil')
    실패      cer > 0.3        (완전 오인식)

사용법:
    좌측 사이드바에서 실험군 + 확대 설정
    상단 탭: 성공 / 부분실패 / 실패 (+ 개수)
    '불일치 하이라이트' ON: GT-Pred 다른 문자를 색으로 표시
"""
import sys
from pathlib import Path
import csv

import streamlit as st
from PIL import Image

# ============================================================
# 경로 설정
# ============================================================
REPO = Path(__file__).resolve().parent.parent        # doctor_ocr_v3_1
EVAL = REPO / "evaluate"                              # v3_1 자체 평가 결과/지표

# 사용 가능한 결과 CSV: 이름 -> 절대경로
AVAILABLE = {
    "v3_1 (Attention, 최종)": EVAL / "result_v3_1_clean.csv",
    "v3 exp2_clean (CTC)": EVAL / "result_exp2_clean.csv",
    "v3 exp1_clean (원본)": EVAL / "result_exp1_clean.csv",
    "v3 exp3_clean (합성)": EVAL / "result_exp3_clean.csv",
}
# 존재하는 것만
AVAILABLE = {k: v for k, v in AVAILABLE.items() if v.exists()}

PARTIAL_CEIL = 0.3          # 부분실패 기준 (1 - CER)


# ============================================================
# 데이터 로드
# ============================================================
def load_results(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cer_f"] = float(r.get("cer", 0) or 0)
        if r["cer_f"] == 0:
            r["cat"] = "성공"
        elif r["cer_f"] <= PARTIAL_CEIL:
            r["cat"] = "부분실패"
        else:
            r["cat"] = "실패"
    return rows


def cat_icon(cat):
    return {"성공": "✅", "부분실패": "🟡", "실패": "❌"}[cat]


# ============================================================
# GT-Pred 문자 하이라이트 (HTML)
# ============================================================
def diff_html(gt: str, pred: str):
    """문자 단위로 다른 글자만 빨간 배경 표시"""
    parts = []
    for a, b in zip(gt, pred):
        if a == b:
            parts.append(f"<span>{b}</span>")
        else:
            parts.append(f"<span style='background:#ff4444;color:white;'>{b}</span>")
    if len(pred) > len(gt):
        for ch in pred[len(gt):]:
            parts.append(f"<span style='background:#ff4444;color:white;'>{ch}</span>")
    elif len(gt) > len(pred):
        parts.append("<span style='background:#ff8800;color:white;'>…</span>")
    return "".join(parts)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="DoctorOcr 결과 뷰어", layout="wide")
st.title("🧪 DoctorOcr — 정성 분석 결과 뷰어")

with st.sidebar:
    st.header("설정")
    exp_key = st.selectbox("실험군", list(AVAILABLE.keys()))
    csv_path = AVAILABLE[exp_key]
    zoom = st.slider("이미지 크기 (px)", 120, 500, 220, 20)
    show_hint = st.toggle("GT-Pred 문자 하이라이트", value=True)
    per_page = st.selectbox("페이지당 개수", [20, 50, 100], index=0)
    st.caption(f"분류: cer=0 성공 / 0<cer≤{PARTIAL_CEIL} 부분실패 / >{PARTIAL_CEIL} 실패")

rows = load_results(csv_path)
st.caption(f"총 {len(rows)}개 — {exp_key}")

# ---- 카테고리 탭 ----
cats = ["성공", "부분실패", "실패"]
counts = {c: sum(1 for r in rows if r["cat"] == c) for c in cats}
tabs = st.tabs([f"{cat_icon(c)} {c} ({counts[c]})" for c in cats])

col_meta = ["path", "label", "group", "cer", "cer_f"]

for tab, cat in zip(tabs, cats):
    with tab:
        subset = [r for r in rows if r["cat"] == cat]

        # 추가 필터: 빈도그룹, 텍스트 검색
        c1, c2, c3 = st.columns([1, 1, 2])
        groups = ["전체"] + sorted({r["group"] for r in subset if "group" in r})
        g = c1.selectbox("빈도그룹", groups, key=f"g_{cat}")
        q = c2.text_input("라벨/예측 검색", "", key=f"q_{cat}")

        filt = subset
        if g != "전체":
            filt = [r for r in filt if r.get("group") == g]
        if q:
            filt = [r for r in filt if q.lower() in (r["true"] + " " + r["pred"]).lower()]

        c3.write(f"**{len(filt)}개 표시** / 전체 {len(subset)}")

        if not filt:
            st.info("조건에 맞는 결과가 없습니다.")
            continue

        # 페이지네이션
        total_pages = max(1, (len(filt) + per_page - 1) // per_page)
        page = st.number_input("페이지", 1, total_pages, 1, key=f"p_{cat}")
        start = (page - 1) * per_page
        page_rows = filt[start:start + per_page]

        # 3열 그리드
        for i in range(0, len(page_rows), 3):
            cols = st.columns(3)
            for col, r in zip(cols, page_rows[i:i + 3]):
                with col:
                    img_path = Path(r["path"])
                    if img_path.exists():
                        img = Image.open(img_path)
                        st.image(img, width=zoom, use_container_width=False)
                    else:
                        st.warning("이미지 없음")
                    st.markdown(f"**GT:** `{r['true']}`")
                    if show_hint:
                        st.markdown(
                            f"**Pred:** {diff_html(r['true'], r['pred'])}",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Pred:** `{r['pred']}`")
                    st.caption(
                        f"CER {r['cer_f']:.2%} · {r.get('group','-')} · {img_path.name}")
                    st.markdown("---")
