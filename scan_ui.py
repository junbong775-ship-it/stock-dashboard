"""스캐너 공용 UI — 단계별 탈락 깔때기(모바일 대응).

봉봉 스캐너와 급등주 스캐너가 동일한 깔때기 패널을 쓰도록 한 곳에 모았다.
모든 동적 문자열은 unsafe_allow_html 전에 escape 한다.

※ 장 세션 배지/시장 라벨 바(`render_session_bar` 등)는 제거됐다 — 시장 선택은
   봉봉 스캐너 탭의 단일 라디오 선택기(🇺🇸 미국 / 🇰🇷 한국)만 사용한다.
"""
from __future__ import annotations

import html as _html

import streamlit as st

# 단계 종류별 색
_KIND_COLOR = {
    "total":  "#5C6BC0",   # 전체/입력
    "reject": "#7A4A5C",   # 탈락
    "pass":   "#00C851",   # 통과
    "info":   "#3A3A5C",   # 중간 정보
}


def render_stage_funnel(
    stages: list[tuple[str, int, str]],
    *,
    title: str = "단계별 필터 현황",
    elapsed: float | None = None,
    container=None,
) -> None:
    """단계별 깔때기를 세로 리스트로 렌더(모바일에서 한 줄씩 보임).

    stages: (라벨, 개수, kind) 리스트. kind ∈ {'total','reject','pass','info'}.
    elapsed: 소요 시간(초). 주어지면 헤더에 표시.
    container: st.empty() 등 렌더 대상(없으면 st 직접). 진행 중 갱신에 사용.
    """
    sink = container if container is not None else st
    head = _html.escape(title)
    if elapsed is not None:
        head += f" · {elapsed:.1f}초"
    rows = [
        '<div style="background:#1A1A2E;border:1px solid #2E2E4E;border-radius:10px;'
        'padding:12px 14px;margin:6px 0 14px 0;">'
        f'<div style="color:#C5CAE9;font-size:0.9rem;font-weight:700;margin-bottom:8px;">'
        f'🔎 {head}</div>'
    ]

    # 깔때기 진행: 'total' 개수에서 시작해 'reject' 단계마다 차감.
    # 각 reject 필터에 대해 진입(차감 전 잔여)·통과(차감 후 잔여)·탈락(이번 제거)을 표기.
    # → 탈락=0 이어도 '진입>0, 통과=진입' 으로 보이므로 "필터 미실행"과 명확히 구분된다.
    remaining: int | None = None
    for label, count, kind in stages:
        color = _KIND_COLOR.get(kind, "#3A3A5C")
        lab = _html.escape(str(label))

        if kind == "total":
            remaining = count
            rows.append(
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                'padding:5px 8px;margin:3px 0;border-radius:7px;background:#13131F;'
                f'border-left:4px solid {color};">'
                f'<span style="color:#B5B5D0;font-size:0.85rem;">{lab}</span>'
                f'<span style="color:#E0E0FF;font-weight:800;font-size:1.05rem;">{count:,}</span>'
                '</div>'
            )
            continue

        if kind == "reject":
            entered = remaining if remaining is not None else count
            rejected = count
            passed = max(entered - rejected, 0)
            remaining = passed
            # 탈락 0 = 회색(이 필터를 모두 통과), 탈락>0 = 강조색
            rej_col = "#7A4A5C" if rejected > 0 else "#3A3A5C"
            rej_emph = "font-weight:700;" if rejected > 0 else "color:#6E6E8E;"
            rows.append(
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                'padding:5px 8px;margin:3px 0;border-radius:7px;background:#13131F;'
                f'border-left:4px solid {rej_col};">'
                f'<span style="color:#B5B5D0;font-size:0.85rem;">{lab}</span>'
                '<span style="font-size:0.8rem;display:flex;gap:10px;align-items:center;">'
                f'<span style="color:#7E84B0;">진입 {entered:,}</span>'
                f'<span style="color:#7E84B0;">통과 {passed:,}</span>'
                f'<span style="color:#E0A0B8;{rej_emph}">탈락 {rejected:,}</span>'
                '</span>'
                '</div>'
            )
            continue

        # pass / info 등은 개수만 표기
        emphasis = "font-weight:800;font-size:1.05rem;" if kind == "pass" else "font-weight:600;"
        rows.append(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'padding:5px 8px;margin:3px 0;border-radius:7px;background:#13131F;'
            f'border-left:4px solid {color};">'
            f'<span style="color:#B5B5D0;font-size:0.85rem;">{lab}</span>'
            f'<span style="color:#E0E0FF;{emphasis}">{count:,}</span>'
            '</div>'
        )
    rows.append('</div>')
    sink.markdown("".join(rows), unsafe_allow_html=True)
