"""순차 실행 헬퍼 (스레드 미사용).

Streamlit Community Cloud 는 가용 메모리가 작고(≈1GB) 스레드 생성이 제한될 수 있어
(`RuntimeError: can't start new thread`) 백그라운드 스레드/스레드풀을 **일절 쓰지 않는다**.

`gather_parallel` 은 기존 호출부와의 호환을 위해 이름을 유지하되 **내부적으로 순차
실행**한다. 전체 마감시한(`deadline_sec`)을 지나면 남은 항목은 `fn` 호출 없이
`default` 로 채워 **결과 길이를 입력과 동일하게 보존**한다(스킵하지 않음).

스레드/풀을 생성하지 않으므로 "can't start new thread" 가 원천적으로 발생할 수 없다.
호출부의 무거운 I/O(뉴스 RSS 등)는 `@st.cache_data` 로 캐시되어 반복 스캔 시
순차 실행이어도 빠르게 끝난다.
"""
from __future__ import annotations

import time


def gather_parallel(fn, items, *, max_workers: int = 8,
                    deadline_sec: float = 45.0, default=None) -> list[tuple]:
    """``items`` 각각에 ``fn`` 을 **순차** 적용하고 ``(item, result)`` 리스트를 반환한다.

    - 스레드/프로세스를 생성하지 않는다(Cloud 안정성: "can't start new thread" 차단).
    - ``deadline_sec`` 를 넘기면 남은 항목은 ``fn`` 호출 없이 ``default`` 로 채운다.
    - 개별 ``fn`` 예외는 삼켜 ``default`` 로 대체한다(앱 중단 방지).
    - ``max_workers`` 는 과거 호환용 인자이며 무시된다.
    """
    items = list(items)
    if not items:
        return []
    out: list[tuple] = []
    deadline = time.time() + deadline_sec
    for it in items:
        if time.time() >= deadline:
            out.append((it, default))
            continue
        try:
            out.append((it, fn(it)))
        except Exception:
            out.append((it, default))
    return out
