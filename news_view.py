"""Shared news rendering: Korean-first cards used by every tab that shows news
(종목 상세분석 · 봉봉/급등주/미래10배주 스캐너) so the behaviour is identical everywhere.

Each card shows, in Korean, without any click:
- the translated headline (제목) — clicking it opens the ORIGINAL article in a new tab
- a short 2~3 line Korean summary (요약, from `summary_ko`)
- 긍정/중립/부정 sentiment + 영향도 점수
- 출처(source) + 날짜(date)

There is no in-app article-body view: free keyless translators block this server's
IP and body extraction is unreliable, so the headline links straight to the source
for full details.
"""
from __future__ import annotations

import datetime as _dt
import html as _html

import streamlit as st

import news_sentiment as _sent

_counter = {'n': 0}


def reset_keys() -> None:
    """Reset the per-run widget-key counter (call once at the start of each run).

    Retained for app.py compatibility; cards no longer create widgets, but the
    helper is kept so existing call sites keep working.
    """
    _counter['n'] = 0


def _next_key(prefix: str = 'newsbtn') -> str:
    _counter['n'] += 1
    return f"{prefix}_{_counter['n']}"


def _fmt_date(published) -> str:
    if published is None:
        return ''
    if getattr(published, 'tzinfo', None) is None:
        try:
            published = published.replace(tzinfo=_dt.timezone.utc)
        except Exception:
            return ''
    kst  = published.astimezone(_dt.timezone(_dt.timedelta(hours=9)))
    now  = _dt.datetime.now(_dt.timezone.utc)
    secs = (now - published).total_seconds()
    if secs < 3600:
        rel = f"{max(1, int(secs // 60))}분 전"
    elif secs < 86400:
        rel = f"{int(secs // 3600)}시간 전"
    else:
        rel = f"{int(secs // 86400)}일 전"
    return f"{kst:%Y-%m-%d %H:%M} (KST) · {rel}"


def article_impact(sent_score) -> int:
    """Per-article 0-100 impact score from a [-1, 1] sentiment score."""
    try:
        s = float(sent_score)
    except Exception:
        s = 0.0
    return max(0, min(100, int(round(50 + s * 50))))


def _render_card(n: dict) -> None:
    # Headline is the Korean-synthesized one ONLY (never the raw English title).
    # We deliberately do NOT fall back to 'translated'/'original' here: if
    # headline_ko is somehow missing we render the (always-Korean) summary card
    # alone, so an English-only headline can never appear.
    headline = (n.get('headline_ko') or '').strip()
    original = str(n.get('original', '') or '').strip()
    summary  = n.get('summary_ko', '') or ''
    if not headline and not summary:
        return
    s_emoji, s_label, color = _sent.label_badge(n.get('sentiment', 'neutral'))
    impact   = article_impact(n.get('sent_score', 0.0))
    date_str = _fmt_date(n.get('published'))
    source   = n.get('source', '') or ''
    url      = str(n.get('url', '') or '')

    icol  = '#2ecc71' if impact >= 60 else '#e74c3c' if impact < 40 else '#9aa0b5'
    badge = (f'<span style="font-size:0.72rem;font-weight:700;color:{color};">{s_emoji} {s_label}</span>'
             f'<span style="font-size:0.72rem;font-weight:700;color:{icol};margin-left:8px;">영향도 {impact}/100</span>')

    # Clickable Korean headline → original article (new tab). Falls back to plain
    # text when no URL is available.
    title_html = ''
    if headline:
        safe_title = _html.escape(headline)
        if url.startswith(('http://', 'https://')):
            title_html = (f'<a href="{_html.escape(url)}" target="_blank" rel="noopener noreferrer" '
                          f'style="font-size:0.92rem;font-weight:700;line-height:1.35;color:#DCE3F0;'
                          f'text-decoration:none;">{safe_title} <span style="color:{color};font-size:0.78rem;">↗</span></a>')
        else:
            title_html = (f'<span style="font-size:0.92rem;font-weight:700;line-height:1.35;'
                          f'color:#DCE3F0;">{safe_title}</span>')
        title_html = f'<div style="margin:2px 0;">{title_html}</div>'

    # Original (English) title shown only as a small, muted secondary line so the
    # reader can see the real source title — never as the primary headline.
    orig_html = ''
    if original and original != headline:
        orig_html = (f'<div style="font-size:0.7rem;line-height:1.3;color:#7C8398;'
                     f'margin:1px 0 2px;">원문: {_html.escape(original)}</div>')

    summary_html = ''
    if summary:
        lines = ''.join(
            f'<div style="font-size:0.78rem;line-height:1.45;color:#AEB6CC;">{_html.escape(ln)}</div>'
            for ln in str(summary).split('\n') if ln.strip()
        )
        summary_html = f'<div style="margin:5px 0 2px;">{lines}</div>'

    meta_txt  = (f"🏷️ {source} · " if source else '') + f"📅 {date_str}"
    safe_meta = _html.escape(meta_txt)

    st.markdown(
        f'<div style="border-left:3px solid {color};background:#1B1B2E;border-radius:6px;'
        f'padding:9px 12px;margin:8px 0;">'
        f'<div style="margin-bottom:3px;">{badge}</div>'
        f'{title_html}'
        f'{orig_html}'
        f'{summary_html}'
        f'<div style="font-size:0.69rem;color:#8A90A6;margin-top:4px;">{safe_meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_news_section(news: list, key_prefix: str = 'news',
                        show_overall: bool = True) -> None:
    """Render the overall sentiment header + one Korean card per article.

    Each card shows the Korean headline (link to the original), a short Korean
    summary, sentiment and impact. Safe to call with an empty list. `key_prefix`
    is accepted for backward compatibility but no longer used (cards have no
    widgets).
    """
    if not news:
        st.caption("최근 3일 내 관련 뉴스 없음")
        return
    if show_overall:
        o_score          = _sent.overall_score(news)
        o_emoji, o_label = _sent.overall_label(o_score)
        o_color          = _sent.overall_color(o_score)
        st.markdown(
            f'<div style="background:{o_color}22;border:1px solid {o_color}66;'
            f'border-radius:8px;padding:8px 11px;margin-bottom:6px;display:flex;'
            f'align-items:center;justify-content:space-between;gap:8px;">'
            f'<span style="font-size:0.82rem;color:#B8C0D6;">종합 뉴스 감성</span>'
            f'<span style="font-size:0.95rem;font-weight:700;color:{o_color};">'
            f'{o_emoji} {o_label} ({o_score}/100)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    for n_item in news:
        try:
            _render_card(n_item)
        except Exception:
            pass
