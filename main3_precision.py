"""main3.py の押し目判定を厳格化して実行する本番エントリーポイント。"""

import sys

import numpy as np
import pandas as pd

import main3 as core

SMA_SHORT_PERIOD = 25
SMA_LONG_PERIOD = 75
SMA_LONG_SLOPE_DAYS = 10
PASS_REASON = "押し目反発"


def compute_one_ticker(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    window_days=60,
):
    """
    main3.py の既存条件に、次の4条件を追加する。

    1. 押し目安値を付けた日は最新日より前
    2. 最新終値が前日終値を上回る
    3. SMA25 > SMA75
    4. SMA75が10営業日前より上
    """
    ticker = getattr(close_s, "name", "")
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
            return None

        close_all = price["Close"]
        look = price.iloc[-window_days:]
        if look.empty:
            return None

        peak_idx = look["High"].idxmax()
        peak_val = float(look.loc[peak_idx, "High"])

        after_peak = look.loc[look.index > peak_idx, "Low"]
        if after_peak.empty:
            return None
        pull_idx = after_peak.idxmin()
        pull_val = float(after_peak.loc[pull_idx])

        latest_idx = close_all.index[-1]
        latest_val = float(close_all.iloc[-1])
        prev_val = float(close_all.iloc[-2])

        # 当日急落の安値を、その日のうちに押し目認定しない。
        if pull_idx >= latest_idx:
            return None

        # 終値ベースの反発を確認する。
        rebound_confirmed = latest_val > prev_val
        if not rebound_confirmed:
            return None

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

        trend_confirmed = (
            np.isfinite(sma25_last)
            and np.isfinite(sma75_last)
            and np.isfinite(sma75_previous)
            and sma25_last > sma75_last
            and sma75_last > sma75_previous
        )
        if not trend_confirmed:
            return None

        drop_pct = (1.0 - latest_val / peak_val) * 100.0
        expected_rise_pct = (peak_val / latest_val - 1.0) * 100.0
        delta_from_pull_pct = (latest_val / pull_val - 1.0) * 100.0

        if np.isclose(latest_val, pull_val, rtol=0.0, atol=1e-6):
            return None

        within_band = (
            latest_val > pull_val
            and latest_val <= pull_val * core.WITHIN_UPPER
        )

        pass_reason = None
        if within_band:
            pass_reason = PASS_REASON
        elif core.USE_RETURN_OR and expected_rise_pct >= core.EXP_OR:
            pass_reason = core.PASS_REASON_EXP

        if not (
            drop_pct <= core.DROP_MAX
            and expected_rise_pct >= core.EXPECTED_RISE_MIN
            and pass_reason is not None
        ):
            return None

        if core.WEEKLY_MA2_FILTER and core._weekly_ma2_is_down(close_all):
            return None

        return {
            "Ticker": ticker,
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
            "Rebound_Confirmed": rebound_confirmed,
            "Trend_Confirmed": trend_confirmed,
            "Within_(pull, +2%]": within_band,
            "OR_Return_ge_5%": (
                core.USE_RETURN_OR and expected_rise_pct >= core.EXP_OR
            ),
            "Pass_Reason": pass_reason,
            "Window": window_days,
        }
    except Exception as exc:
        print(
            f"[WARN] precision compute failed for {ticker}: {exc}",
            file=sys.stderr,
        )
        return None


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
            "該当銘柄はありませんでした。"
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
    # main3内のスクリーニングと通知だけを厳格版へ差し替える。
    core.compute_one_ticker = compute_one_ticker
    core.notify = notify
    core.main()


if __name__ == "__main__":
    main()
