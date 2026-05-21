import logging
import pandas as pd
import pandas_market_calendars as mcal
from config.settings import TICKER_CALENDAR, HORIZONS

logger = logging.getLogger(__name__)

_cal_cache: dict = {}


def _get_cal(name: str):
    if name not in _cal_cache:
        _cal_cache[name] = mcal.get_calendar(name)
    return _cal_cache[name]


def target_dates(base: pd.Timestamp, ticker: str) -> dict[int, pd.Timestamp]:
    """Return {1: date_d1, 2: date_d2, 3: date_d3} respecting the ticker's market calendar."""
    cal_name = TICKER_CALENDAR.get(ticker, "NYSE")
    try:
        cal   = _get_cal(cal_name)
        start = (base + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (base + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        sched = cal.schedule(start_date=start, end_date=end)
        result = {}
        for n in HORIZONS:
            if len(sched) >= n:
                result[n] = pd.Timestamp(sched.index[n - 1].date())
            else:
                result[n] = base + pd.offsets.BDay(n)
        return result
    except Exception as e:
        logger.warning("Calendário %s falhou (%s) — usando BDay", cal_name, e)
        return {n: base + pd.offsets.BDay(n) for n in HORIZONS}
