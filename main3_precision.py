"""押し目候補を黄色/緑の2段階で判定し、診断・OHLCVを保存する本番エントリーポイント。"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import main3 as core

SMA_SHORT_PERIOD = 25
SMA_LONG_PERIOD = 75
SMA_LONG_SLOPE_DAYS = 10

APPROACH_BAND_PCT = float(os.getenv("APPROACH_BAND_PCT", "2.0"))
CONFIRMED_BAND_PCT = float(os.getenv("CONFIRMED_BAND_PCT", "5.0"))
OHLCV_LOOKBACK_DAYS = int(os.getenv("OHLCV_LOOKBACK_DAYS", "90"))

SIGNAL_YELLOW = "YELLOW"
SIGNAL_GREEN = "GREEN"
PASS_REASON_YELLOW = "🟡押し目接近"
PASS_REASON_GREEN = "🟢反発確認"

REPORT_DIR = os.getenv("PULLBACK_REPORT_DIR", "reports")
DIAGNOSTICS_CSV = os.path.join(REPORT_DIR, "pullback_diagnostics_latest.csv")
CANDIDATES_CSV = os.path.join(REPORT_DIR, "pullback_candidates_latest.csv")
REJECTION_SUMMARY_CSV = os.path.join(REPORT_DIR, "pullback_rejection_summary_latest.csv")
NEAR_MISS_CSV = os.path.join(REPORT_DIR, "pullback_near_miss_latest.csv")
OHLCV_CSV = os.path.join(REPORT_DIR, "pullback_ohlcv_latest.csv")

CONDITION_COLUMNS = [
    "SMA25_Above_SMA75",
    "SMA75_Up",
    "Drop_OK",
    "Return_OK",
    "Green_Band_OK",
    "Pull_Before_Latest",
    "Rebound_Confirmed",
]


def _empty_row(ticker: str, window_days: int) -> Dict:
    row = {
        "Ticker": ticker,
        "Name": core.TICKER_NAME_MAP.get(ticker, ""),
        "Window": window_days,
        "Passed": False,
        "Signal_Level": "",
        "Signal_Rank": 0,
        "Pass_Reason": "",
        "Reject_Reason": "",
        "Failed_Conditions": "",
        "Conditions_Passed": 0,
        "Weekly_MA2_Warning": False,
    }
    for col in CONDITION_COLUMNS:
        row[col] = False
    row["Yellow_Band_OK"] = False
    row["Weekly_MA2_OK"] = True
    return row


def evaluate_one_ticker(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    window_days: int = 60,
) -> Dict:
    """1銘柄・1期間を評価し、黄色・緑・除外のいずれでも診断行を返す。"""
    ticker = str(getattr(close_s, "name", "") or "")
    row = _empty_row(ticker, window_days)

    try:
        price = pd.concat(
            [
                close_s.rename("Close"),
                high_s.rename("High"),
                low_s.rename("Low"),
            ],
            axis=1,
        ).dropna()

        required_len = max(
            window_days + 2,
            SMA_LONG_PERIOD + SMA_LONG_SLOPE_DAYS + 1,
        )
        if len(price) < required_len:
            row["Reject_Reason"] = "データ不足"
            row["Data_Rows"] = len(price)
            return row

        close_all = price["Close"]
        look = price.iloc[-window_days:]
        if look.empty:
            row["Reject_Reason"] = "対象期間データなし"
            return row

        peak_idx = look["High"].idxmax()
        peak_val = float(look.loc[peak_idx, "High"])

        after_peak = look.loc[look.index > peak_idx, "Low"]
        if after_peak.empty:
            row["Reject_Reason"] = "ピーク後データなし"
            row["Peak_Date"] = peak_idx.date()
            row["Peak_High"] = round(peak_val, 2)
            return row

        pull_idx = after_peak.idxmin()
        pull_val = float(after_peak.loc[pull_idx])

        latest_idx = close_all.index[-1]
        latest_val = float(close_all.iloc[-1])
        prev_val = float(close_all.iloc[-2])

        sma25 = close_all.rolling(
            SMA_SHORT_PERIOD,
            min_periods=SMA_SHORT_PERIOD,
        ).mean()
        sma75 = close_all.rolling(
            SMA_LONG_PERIOD,
            min_periods=SMA_LONG_PERIOD,
        ).mean()

        sma25_last = float(sma25.iloc[-1])
        sma75_last = float(sma75.iloc[-1])
        sma75_previous = float(sma75.iloc[-1 - SMA_LONG_SLOPE_DAYS])

        drop_pct = (1.0 - latest_val / peak_val) * 100.0
        expected_rise_pct = (peak_val / latest_val - 1.0) * 100.0
        delta_from_pull_pct = (latest_val / pull_val - 1.0) * 100.0

        pull_before_latest = bool(pull_idx < latest_idx)
        rebound_confirmed = bool(latest_val > prev_val)
        sma25_above_sma75 = bool(
            np.isfinite(sma25_last)
            and np.isfinite(sma75_last)
            and sma25_last > sma75_last
        )
        sma75_up = bool(
            np.isfinite(sma75_last)
            and np.isfinite(sma75_previous)
            and sma75_last > sma75_previous
        )
        drop_ok = bool(drop_pct <= core.DROP_MAX)
        return_ok = bool(expected_rise_pct >= core.EXPECTED_RISE_MIN)

        yellow_upper = 1.0 + APPROACH_BAND_PCT / 100.0
        green_upper = 1.0 + CONFIRMED_BAND_PCT / 100.0
        yellow_band_ok = bool(
            latest_val > pull_val and latest_val <= pull_val * yellow_upper
        )
        green_band_ok = bool(
            latest_val > pull_val and latest_val <= pull_val * green_upper
        )

        weekly_ma2_ok = bool(not core._weekly_ma2_is_down(close_all))
        weekly_warning = not weekly_ma2_ok

        trend_ok = sma25_above_sma75 and sma75_up
        base_ok = drop_ok and return_ok

        green_signal = bool(
            trend_ok
            and base_ok
            and green_band_ok
            and pull_before_latest
            and rebound_confirmed
        )
        yellow_signal = bool(
            not green_signal
            and trend_ok
            and base_ok
            and yellow_band_ok
        )

        if green_signal:
            signal_level = SIGNAL_GREEN
            signal_rank = 2
            pass_reason = PASS_REASON_GREEN
        elif yellow_signal:
            signal_level = SIGNAL_YELLOW
            signal_rank = 1
            pass_reason = PASS_REASON_YELLOW
        else:
            signal_level = ""
            signal_rank = 0
            pass_reason = ""

        row.update(
            {
                "Peak_Date": peak_idx.date(),
                "Peak_High": round(peak_val, 2),
                "Pullback_Date": pull_idx.date(),
                "Pullback_Low": round(pull_val, 2),
                "Latest_Date": latest_idx.date(),
                "Latest_Close": round(latest_val, 2),
                "Prev_Close": round(prev_val, 2),
                "Return_%": round(expected_rise_pct, 2),
                "Drop_From_Peak_%": round(drop_pct, 2),
                "Delta_from_Pull_%": round(delta_from_pull_pct, 2),
                "SMA25": round(sma25_last, 2),
                "SMA75": round(sma75_last, 2),
                "SMA75_10d_Ago": round(sma75_previous, 2),
                "Pull_Before_Latest": pull_before_latest,
                "Rebound_Confirmed": rebound_confirmed,
                "SMA25_Above_SMA75": sma25_above_sma75,
                "SMA75_Up": sma75_up,
                "Drop_OK": drop_ok,
                "Return_OK": return_ok,
                "Yellow_Band_OK": yellow_band_ok,
                "Green_Band_OK": green_band_ok,
                "Weekly_MA2_OK": weekly_ma2_ok,
                "Weekly_MA2_Warning": weekly_warning,
                f"Within_(pull, +{APPROACH_BAND_PCT:g}%]": yellow_band_ok,
                f"Within_(pull, +{CONFIRMED_BAND_PCT:g}%]": green_band_ok,
                "Signal_Level": signal_level,
                "Signal_Rank": signal_rank,
                "Passed": bool(signal_level),
                "Pass_Reason": pass_reason,
            }
        )

        failed_labels: List[str] = []
        if not sma25_above_sma75:
            failed_labels.append("SMA25<=SMA75")
        if not sma75_up:
            failed_labels.append("SMA75下向き")
        if not drop_ok:
            failed_labels.append(f"Drop>{core.DROP_MAX:.0f}%")
        if not return_ok:
            failed_labels.append(f"Return<{core.EXPECTED_RISE_MIN:.0f}%")
        if not green_band_ok:
            failed_labels.append(f"押し目安値+{CONFIRMED_BAND_PCT:g}%外")
        elif not yellow_band_ok and not (pull_before_latest and rebound_confirmed):
            failed_labels.append(
                f"反発未確認かつ押し目安値+{APPROACH_BAND_PCT:g}%外"
            )
        if green_band_ok and not pull_before_latest:
            failed_labels.append("押し目日が最新日")
        if green_band_ok and not rebound_confirmed:
            failed_labels.append("前日終値超えなし")

        row["Conditions_Passed"] = sum(bool(row[c]) for c in CONDITION_COLUMNS)
        row["Failed_Conditions"] = " | ".join(failed_labels)
        row["Reject_Reason"] = failed_labels[0] if failed_labels else ""

        if row["Passed"]:
            row["Reject_Reason"] = ""
            row["Failed_Conditions"] = ""

        return row

    except Exception as exc:
        row["Reject_Reason"] = f"計算エラー: {type(exc).__name__}"
        row["Error"] = str(exc)
        print(
            f"[WARN] precision evaluation failed for {ticker}: {exc}",
            file=sys.stderr,
        )
        return row


def compute_one_ticker(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    window_days: int = 60,
) -> Optional[Dict]:
    """既存スクリーナー互換。黄色または緑の場合だけ辞書を返す。"""
    row = evaluate_one_ticker(close_s, high_s, low_s, window_days)
    return row if row.get("Passed") else None


def build_diagnostics(raw_df: pd.DataFrame, tickers) -> pd.DataFrame:
    close, high, low = core.extract_price_frames(raw_df, tickers)
    rows = []

    for ticker in tickers:
        if ticker not in close.columns:
            for window_days in core.SCREEN_WINDOWS:
                row = _empty_row(ticker, window_days)
                row["Reject_Reason"] = "価格データなし"
                rows.append(row)
            continue

        for window_days in core.SCREEN_WINDOWS:
            rows.append(
                evaluate_one_ticker(
                    close[ticker],
                    high[ticker],
                    low[ticker],
                    window_days,
                )
            )

    return pd.DataFrame(rows)


def select_candidates(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics is None or diagnostics.empty or "Passed" not in diagnostics.columns:
        return pd.DataFrame()

    passed = diagnostics[diagnostics["Passed"] == True].copy()
    if passed.empty:
        return pd.DataFrame()

    passed["Return_%"] = pd.to_numeric(passed["Return_%"], errors="coerce")
    passed["Signal_Rank"] = pd.to_numeric(
        passed["Signal_Rank"], errors="coerce"
    ).fillna(0)
    passed["Pullback_Date_Sort"] = pd.to_datetime(
        passed["Pullback_Date"], errors="coerce"
    )

    best = (
        passed.sort_values(
            ["Ticker", "Signal_Rank", "Pullback_Date_Sort", "Return_%"],
            ascending=[True, False, False, False],
        )
        .groupby("Ticker", as_index=False)
        .first()
        .sort_values(
            ["Signal_Rank", "Return_%"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return best.drop(columns=["Pullback_Date_Sort"], errors="ignore")


def build_ohlcv_history(
    raw_df: pd.DataFrame,
    tickers,
    lookback_days: int = OHLCV_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """全銘柄の最新OHLCVを縦持ちCSV用に整形する。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    frames = []

    for ticker in tickers:
        try:
            if isinstance(raw_df.columns, pd.MultiIndex):
                available = [
                    field for field in fields if (field, ticker) in raw_df.columns
                ]
                if not available:
                    continue
                frame = raw_df.loc[:, [(field, ticker) for field in available]].copy()
                frame.columns = available
            else:
                available = [field for field in fields if field in raw_df.columns]
                if not available:
                    continue
                frame = raw_df[available].copy()

            required_ohlc = [
                field for field in ["Open", "High", "Low", "Close"]
                if field in frame.columns
            ]
            frame = frame.dropna(subset=required_ohlc)
            if frame.empty:
                continue

            frame = frame.tail(lookback_days).reset_index()
            date_col = frame.columns[0]
            frame = frame.rename(
                columns={
                    date_col: "Date",
                    "Adj Close": "Adj_Close",
                }
            )
            frame.insert(1, "Ticker", ticker)
            frame.insert(2, "Name", core.TICKER_NAME_MAP.get(ticker, ""))
            frames.append(frame)
        except Exception as exc:
            print(f"[WARN] OHLCV build failed for {ticker}: {exc}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    preferred = [
        "Date",
        "Ticker",
        "Name",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]
    return out[[col for col in preferred if col in out.columns]]


def save_reports(
    diagnostics: pd.DataFrame,
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    candidates.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    ohlcv.to_csv(OHLCV_CSV, index=False, encoding="utf-8-sig")

    if diagnostics.empty:
        summary = pd.DataFrame(columns=["Reject_Reason", "Count"])
        near_miss = diagnostics.copy()
    else:
        summary = (
            diagnostics.loc[diagnostics["Passed"] != True, "Reject_Reason"]
            .fillna("不明")
            .replace("", "不明")
            .value_counts()
            .rename_axis("Reject_Reason")
            .reset_index(name="Count")
        )

        near_miss = diagnostics[diagnostics["Passed"] != True].copy()
        near_miss["Conditions_Passed"] = pd.to_numeric(
            near_miss["Conditions_Passed"], errors="coerce"
        ).fillna(0)
        near_miss["Delta_from_Pull_%"] = pd.to_numeric(
            near_miss.get("Delta_from_Pull_%"), errors="coerce"
        )
        near_miss = near_miss.sort_values(
            ["Conditions_Passed", "Delta_from_Pull_%"],
            ascending=[False, True],
            na_position="last",
        ).head(50)

    summary.to_csv(REJECTION_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    near_miss.to_csv(NEAR_MISS_CSV, index=False, encoding="utf-8-sig")

    print(f"[INFO] diagnostics saved: {DIAGNOSTICS_CSV}")
    print(f"[INFO] candidates saved: {CANDIDATES_CSV}")
    print(f"[INFO] rejection summary saved: {REJECTION_SUMMARY_CSV}")
    print(f"[INFO] near misses saved: {NEAR_MISS_CSV}")
    print(f"[INFO] OHLCV saved: {OHLCV_CSV} ({len(ohlcv)} rows)")


def _fp(value, digits=1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _fdate(value) -> str:
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%m/%d")
        return pd.to_datetime(value).strftime("%m/%d")
    except Exception:
        return "-"


def _signal_lines(df: pd.DataFrame, level: str, title: str) -> List[str]:
    subset = df[df["Signal_Level"] == level].copy()
    if subset.empty:
        return [f"{title}: 0銘柄"]

    lines = [f"{title}: {len(subset)}銘柄"]
    for _, row in subset.iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        weekly_note = " ⚠週足MA2↓" if bool(row.get("Weekly_MA2_Warning")) else ""
        lines.append(
            f"{ticker:<8} {name:<8} "
            f"Return:{_fp(row.get('Return_%'), 1):>5}% "
            f"Drop:{_fp(row.get('Drop_From_Peak_%'), 1):>5}% "
            f"ΔPull:{_fp(row.get('Delta_from_Pull_%'), 1):>5}% "
            f"Win:{int(row.get('Window') or 0):>2}d "
            f"Pull:{_fdate(row.get('Pullback_Date'))}"
            f"{weekly_note}"
        )
    return lines


def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    """黄色（接近）と緑（反発確認）を分けてDiscord通知する。"""
    condition_text = (
        f"共通: Drop≤{core.DROP_MAX:.0f}%・Return≥{core.EXPECTED_RISE_MIN:.0f}%・"
        "SMA25>SMA75・SMA75上向き\n"
        f"🟡 接近: 押し目安値+{APPROACH_BAND_PCT:g}%以内（反発未確認も対象）\n"
        f"🟢 確認: 押し目安値+{CONFIRMED_BAND_PCT:g}%以内・"
        "押し目日は前営業日以前・前日終値超え\n"
        "※週足MA2下向きは除外せず⚠表示"
    )

    if df is None or df.empty:
        core.send_long_text(
            f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}\n"
            f"{condition_text}\n"
            "黄色・緑ともに該当銘柄はありませんでした。\n"
            "診断CSV: pullback_diagnostics_latest.csv"
        )
        return

    lines = [
        f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}",
        condition_text,
        "------------------------------",
    ]
    lines.extend(_signal_lines(df, SIGNAL_GREEN, "🟢 反発確認"))
    lines.append("------------------------------")
    lines.extend(_signal_lines(df, SIGNAL_YELLOW, "🟡 押し目接近"))
    core.send_long_text("\n".join(lines))

    if not core.MPF_AVAILABLE:
        print(
            "[INFO] mplfinance not installed; charts will not be generated.",
            file=sys.stderr,
        )
        return

    ordered = df.sort_values(
        ["Signal_Rank", "Return_%"],
        ascending=[False, False],
    ).head(core.CHART_TOP_N)

    for _, row in ordered.iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        reason = str(row.get("Pass_Reason", "")) or "-"

        rsi = core.latest_rsi_from_raw(raw_df, ticker, period=core.RSI_PERIOD)
        rsi_text = "-" if rsi is None or not np.isfinite(rsi) else f"{rsi:.0f}"
        weekly_note = " / ⚠週足MA2↓" if bool(row.get("Weekly_MA2_Warning")) else ""
        description = (
            f"Return: {_fp(row.get('Return_%'), 1)}%  "
            f"Drop: {_fp(row.get('Drop_From_Peak_%'), 1)}%  "
            f"ΔPull: {_fp(row.get('Delta_from_Pull_%'), 1)}%  "
            f"Win: {int(row.get('Window') or 0)}d  "
            f"RSI14: {rsi_text}{weekly_note}\n"
            f"Latest: {_fp(row.get('Latest_Close'), 0)}  "
            f"PullLow: {_fp(row.get('Pullback_Low'), 0)}  "
            f"Peak: {_fp(row.get('Peak_High'), 0)}\n"
            f"SMA25: {_fp(row.get('SMA25'), 0)}  "
            f"SMA75: {_fp(row.get('SMA75'), 0)}"
        )

        image_path = core.save_chart_image_from_raw(raw_df, ticker)
        if image_path:
            core.discord_send_image_file(
                image_path,
                title=f"{reason} {ticker} {name}".strip(),
                description=description,
            )


def main():
    now = core.now_jst()
    if not core.FORCE_RUN and core.is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で強制実行）")
        return

    tickers = core.load_tickers()
    raw = core.fetch_market_data(tickers, lookback_days=core.LOOKBACK_DAYS)

    if raw is None or raw.empty:
        core.send_long_text(
            f"【押し目スクリーニング】{now:%m/%d %H:%M}\nデータ取得失敗"
        )
        return

    diagnostics = build_diagnostics(raw, tickers)
    candidates = select_candidates(diagnostics)
    ohlcv = build_ohlcv_history(raw, tickers)
    save_reports(diagnostics, candidates, ohlcv)
    notify(candidates, raw)


if __name__ == "__main__":
    main()
