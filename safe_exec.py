"""Streamlit Cloud 안전 병렬 실행 헬퍼.

왜 필요한가
-----------
`with ThreadPoolExecutor() as ex:` 블록은 종료 시 `shutdown(wait=True)`를 호출해
**모든 작업이 끝날 때까지 대기**한다. 워커가 외부 I/O(yfinance `.info`, 뉴스 RSS 등)에서
멈추면 스캔이 무한 대기에 빠지고, 그동안 Streamlit이 스크립트를 재실행(rerun)하거나
중지할 수 없어 화면이 '무한 로딩' 상태가 된다(사용자가 탭을 바꾸거나 Cloud 헬스체크가
재실행을 트리거할 때 특히 자주 발생).

또 Streamlit Community Cloud는 가용 메모리가 작아(≈1GB) 워커 수가 많으면 OOM으로
앱이 통째로 재시작(빨간 에러 화면)된다.

`gather_parallel`은 전체 마감시한(deadline) 안에서만 폴링하며, 마감 후 남은 작업을
취소하고 `shutdown(wait=False, cancel_futures=True)`로 **즉시** 정리한다. 파이썬의
블로킹 I/O 스레드는 강제 종료가 불가능하므로(멈춘 스레드는 백그라운드에서 알아서
끝나도록 '버린다'), join/wait를 하지 않는 것이 핵심이다.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor


def gather_parallel(fn, items, *, max_workers: int = 8,
                    deadline_sec: float = 45.0, default=None) -> list[tuple]:
    """``items`` 각각에 ``fn``을 병렬 적용하고 ``(item, result)`` 리스트를 반환한다.

    - 전체 ``deadline_sec`` 안에 끝난 결과만 모으고, 마감/예외로 못 끝낸 항목은
      ``default``로 채워 **결과 길이를 입력과 동일하게 보존**한다(스킵하지 않음).
    - 어떤 경우에도 블록을 무한 대기시키지 않는다(`shutdown(wait=False)`).
    - 제출/실행 단계의 예외도 삼켜 앱이 중단되지 않게 한다.
    """
    items = list(items)
    if not items:
        return []

    results: dict[int, object] = {}
    ex: ThreadPoolExecutor | None = None
    try:
        workers = max(1, min(max_workers, len(items)))
        ex = ThreadPoolExecutor(max_workers=workers)
        fut_to_idx = {ex.submit(fn, it): i for i, it in enumerate(items)}
        pending = set(fut_to_idx)
        deadline = time.time() + deadline_sec
        while pending and time.time() < deadline:
            done = {f for f in pending if f.done()}
            if not done:
                time.sleep(0.05)
                continue
            for f in done:
                idx = fut_to_idx[f]
                try:
                    results[idx] = f.result(timeout=0)
                except Exception:
                    results[idx] = default
            pending -= done
        # 마감 후 남은 작업은 취소(이미 실행 중인 블로킹 스레드는 버린다).
        for f in pending:
            f.cancel()
    except Exception:
        # 제출 단계 등에서의 예외도 앱을 죽이지 않는다.
        pass
    finally:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)

    return [(items[i], results.get(i, default)) for i in range(len(items))]
