"""main3.py の押し目判定を厳格化し、診断CSVも保存する本番エントリーポイント。"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import main3 as core

SMA_SHORT_PERIOD = 25
SMA_LONG_PERIOD = 75
SMA_LONG_SLOPE_DAYS = 10
PASS_REASON = "押し目反発"

REPORT_DIR = os.getenv("PULLBACK_REPORT_DIR", "reports")
DIAGNOSTICS_CSV = os.path.join(REPORT_DIR, "pullback_diagnostics_latest.csv")
CANDIDATES_CSV = os.path.join(REPORT_DIR, "pullback_candidates_latest.csv")
REJECTION_SUMMARY_CSV = os.path.join(REPORT_DIR, "pullback_rejection_summary_latest.csv")
NEAR_MISS_CSV = os.path.join(REPORT_DIR, "pullback_near_miss_latest.csv")

CONDITION_COLUMNS = [
    "Pull_Before_Latest",
    "Rebound_Confirmed",
    "SMA25_Above_SMA75",
    "SMA75_Up",
    "Drop_OK",
    "Return_OK",
    "Band_OK",
    "Weekly_MA2_OK",
]


def _empty_row(ticker: str, window_days: int) -> Dict:
    row = {
        "Ticker": ticker,
        "Name": core.TICKER_NAME_MAP.get(ticker, ""),
        "Window": window_days,
        "Passed": False,
        "Reject_Reason": "",
        "Failed_Conditions": "",
        "Conditions_Passed": 0,
    }
    for col in CONDITION_COLUMNS:
        row[col] = False
    return row


def evaluate_one_ticker(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    window_days: int = 60,
) -> Dict:
    """1銘柄・1期間を評価し、通過・不通過を問わず診断行を返す。"""
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
        within_band = bool(
            latest_val > pull_val
            and latest_val <= pull_val * core.WITHIN_UPPER
        )
        return_or_ok = bool(
            core.USE_RETURN_OR and expected_rise_pct >= core.EXP_OR
        )
        band_ok = bool(within_band or return_or_ok)
        weekly_ma2_ok = bool(
            not core.WEEKLY_MA2_FILTER
            or not core._weekly_ma2_is_down(close_all)
        )

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
                "Band_OK": band_ok,
                "Weekly_MA2_OK": weekly_ma2_ok,
                "Within_(pull, +2%]": within_band,
                "OR_Return_ge_5%": return_or_ok,
            }
        )

        failed_labels: List[str] = []
        checks = [
            ("Pull_Before_Latest", "押し目日が最新日"),
            ("Rebound_Confirmed", "前日終値超えなし"),
            ("SMA25_Above_SMA75", "SMA25<=SMA75"),
            ("SMA75_Up", "SMA75下向き"),
            ("Drop_OK", f"Drop>{core.DROP_MAX:.0f}%"),
            ("Return_OK", f"Return<{core.EXPECTED_RISE_MIN:.0f}%"),
            ("Band_OK", f"押し目安値+{core.PULLBACK_BAND_PCT:.0f}%外"),
            ("Weekly_MA2_OK", "週足MA2下向き"),
        ]
        for col, label in checks:
            if not bool(row[col]):
                failed_labels.append(label)

        row["Conditions_Passed"] = sum(bool(row[c]) for c in CONDITION_COLUMNS)
        row["Failed_Conditions"] = " | ".join(failed_labels)
        row["Reject_Reason"] = failed_labels[0] if failed_labels else ""
        row["Passed"] = not failed_labels
        row["Pass_Reason"] = PASS_REASON if row["Passed"] else ""
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
    """既存スクリーナー互換。通過した場合だけ辞書を返す。"""
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
    return (
        passed.sort_values(["Ticker", "Return_%"], ascending=[True, False])
        .groupby("Ticker", as_index=False)
        .first()
        .sort_values("Return_%", ascending=False)
        .reset_index(drop=True)
    )


def save_reports(diagnostics: pd.DataFrame, candidates: pd.DataFrame) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    diagnostics.to_csv(DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    candidates.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")

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


def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    """追加フィルターとSMA値が分かる形でDiscord通知する。"""
    condition_text = (
        f"Drop≤{core.DROP_MAX:.0f}%・Return≥{core.EXPECTED_RISE_MIN:.0f}%・"
        f"押し目安値+{core.PULLBACK_BAND_PCT:.0f}%以内"
        f"{' or Return≥' + str(int(core.EXP_OR)) + '%' if core.USE_RETURN_OR else ''}・"
        "押し目日は前営業日以前・前日終値超え・"
        "SMA25>SMA75・SMA75上向き・週足MA(2)下向き除外"
    )

    if df is None or df.empty:
        core.send_long_text(
            f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}\n"
            f"条件: {condition_text}\n"
            "該当銘柄はありませんでした。\n"
            "診断CSV: pullback_diagnostics_latest.csv"
        )
        return

    lines = [
        f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}\n"
        f"条件: {condition_text}\n"
        f"抽出: {len(df)} 銘柄\n"
        "------------------------------"
    ]

    def fp(value, digits=1):
        try:
            if pd.isna(value):
                return "-"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def fdate(value):
        try:
            if hasattr(value, "strftime"):
                return value.strftime("%m/%d")
            return pd.to_datetime(value).strftime("%m/%d")
        except Exception:
            return "-"

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        reason = str(row.get("Pass_Reason", "")) or "-"
        lines.append(
            f"{ticker:<8} {name:<8} [{reason}]  "
            f"Return: {fp(row.get('Return_%'), 1):>5}%  "
            f"Drop: {fp(row.get('Drop_From_Peak_%'), 1):>5}%  "
            f"ΔPull: {fp(row.get('Delta_from_Pull_%'), 1):>5}%  "
            f"Win: {int(row.get('Window') or 0):>2}d  "
            f"Pull: {fdate(row.get('Pullback_Date'))}"
        )

    core.send_long_text("\n".join(lines))

    if not core.MPF_AVAILABLE:
        print(
            "[INFO] mplfinance not installed; charts will not be generated.",
            file=sys.stderr,
        )
        return

    for _, row in df.head(core.CHART_TOP_N).iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        reason = str(row.get("Pass_Reason", "")) or "-"

        rsi = core.latest_rsi_from_raw(raw_df, ticker, period=core.RSI_PERIOD)
        rsi_text = "-" if rsi is None or not np.isfinite(rsi) else f"{rsi:.0f}"
        description = (
            f"Return: {fp(row.get('Return_%'), 1)}%  "
            f"Drop: {fp(row.get('Drop_From_Peak_%'), 1)}%  "
            f"ΔPull: {fp(row.get('Delta_from_Pull_%'), 1)}%  "
            f"Win: {int(row.get('Window') or 0)}d  "
            f"RSI14: {rsi_text}\n"
            f"Latest: {fp(row.get('Latest_Close'), 0)}  "
            f"PullLow: {fp(row.get('Pullback_Low'), 0)}  "
            f"Peak: {fp(row.get('Peak_High'), 0)}\n"
            f"SMA25: {fp(row.get('SMA25'), 0)}  "
            f"SMA75: {fp(row.get('SMA75'), 0)}"
        )

        image_path = core.save_chart_image_from_raw(raw_df, ticker)
        if image_path:
            core.discord_send_image_file(
                image_path,
                title=f"{ticker} {name} [{reason}]".strip(),
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
    save_reports(diagnostics, candidates)
    notify(candidates, raw)


if __name__ == "__main__":
    main()
