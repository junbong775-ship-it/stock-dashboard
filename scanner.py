from data_provider import US_TICKERS, KR_TICKERS, KR_NAMES, get_data
from indicators import add_indicators
from ai_engine import calculate_ai_score


def scan_market(market: str, max_spread: float) -> list[dict]:
    if market == 'us':
        ticker_list = [(t, t, 'us') for t in US_TICKERS]
    elif market == 'kr':
        ticker_list = [(c, KR_NAMES.get(c, c), 'kr') for c in KR_TICKERS]
    else:
        ticker_list = ([(t, t, 'us') for t in US_TICKERS] +
                       [(c, KR_NAMES.get(c, c), 'kr') for c in KR_TICKERS])

    results: list[dict] = []
    for code, name, mkt in ticker_list:
        try:
            raw = get_data(code, mkt)
            if raw is None:
                continue
            data = add_indicators(raw)
            if data['spread'] > max_spread:
                continue
            scored = calculate_ai_score(data)
            results.append({
                'code':   code,
                'name':   name,
                'market': mkt,
                **data,
                **scored,
            })
        except Exception:
            continue

    results.sort(key=lambda x: (x.get('ma300_strategy', False), x.get('score', 0)), reverse=True)
    return results
