"""25日線の3営業日連続上向きを必須にした、画像カード専用の押し目スクリーナー。"""

import os
import sys
from typing import Dict, List

import matplotlib as mpl
import numpy as np
import pandas as pd

import main3_precision as app

core = app.core

# GitHub Actionsでは fonts-noto-cjk を導入して使用する。
mpl.rcParams["font.family"] = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "sans-serif",
]
mpl.rcParams["axes.unicode_minus"] = False

SMA25_UP_STREAK_DAYS = 3

CONDITION_COLUMNS = [
    "SMA25_Above_SMA75",
    "SMA75_Up",
    "SMA25_Up_3d",
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
    }
    for column in CONDITION_COLUMNS:
        row[column] = False
    row["Yellow_Band_OK"] = False
    return row


def _is_sma25_up_three_days(sma25: pd.Series) -> bool:
    """25日線が直近3営業日、毎日切り上がっているか判定する。"""
    values = pd.to_numeric(sma25.tail(SMA25_UP_STREAK_DAYS + 1), errors="coerce")
    if len(values) < SMA25_UP_STREAK_DAYS + 1 or values.isna().any():
        return False
    return bool((values.diff().dropna() > 0).all())


def evaluate_one_ticker(
    close_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    window_days: int = 60,
) -> Dict:
    """1銘柄・1期間を、25日線3営業日連続上向き条件付きで評価する。"""
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
            app.SMA_LONG_PERIOD + app.SMA_LONG_SLOPE_DAYS + 1,
            app.SMA_SHORT_PERIOD + SMA25_UP_STREAK_DAYS,
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
            app.SMA_SHORT_PERIOD,
            min_periods=app.SMA_SHORT_PERIOD,
        ).mean()
        sma75 = close_all.rolling(
            app.SMA_LONG_PERIOD,
            min_periods=app.SMA_LONG_PERIOD,
        ).mean()

        sma25_last = float(sma25.iloc[-1])
        sma75_last = float(sma75.iloc[-1])
        sma75_previous = float(sma75.iloc[-1 - app.SMA_LONG_SLOPE_DAYS])
        sma25_up_3d = _is_sma25_up_three_days(sma25)

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

        yellow_upper = 1.0 + app.APPROACH_BAND_PCT / 100.0
        green_upper = 1.0 + app.CONFIRMED_BAND_PCT / 100.0
        yellow_band_ok = bool(
            latest_val > pull_val and latest_val <= pull_val * yellow_upper
        )
        green_band_ok = bool(
            latest_val > pull_val and latest_val <= pull_val * green_upper
        )

        # 黄色・緑ともに、25日線3営業日連続上向きを必須とする。
        trend_ok = sma25_above_sma75 and sma75_up and sma25_up_3d
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
            signal_level = app.SIGNAL_GREEN
            signal_rank = 2
            pass_reason = app.PASS_REASON_GREEN
        elif yellow_signal:
            signal_level = app.SIGNAL_YELLOW
            signal_rank = 1
            pass_reason = app.PASS_REASON_YELLOW
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
                "SMA25_Up_3d": sma25_up_3d,
                "Pull_Before_Latest": pull_before_latest,
                "Rebound_Confirmed": rebound_confirmed,
                "SMA25_Above_SMA75": sma25_above_sma75,
                "SMA75_Up": sma75_up,
                "Drop_OK": drop_ok,
                "Return_OK": return_ok,
                "Yellow_Band_OK": yellow_band_ok,
                "Green_Band_OK": green_band_ok,
                f"Within_(pull, +{app.APPROACH_BAND_PCT:g}%]": yellow_band_ok,
                f"Within_(pull, +{app.CONFIRMED_BAND_PCT:g}%]": green_band_ok,
                "Signal_Level": signal_level,
                "Signal_Rank": signal_rank,
                "Passed": bool(signal_level),
                "Pass_Reason": pass_reason,
            }
        )

        failed_labels: List[str] = []
        if not sma25_above_sma75:
            failed_labels.append("25日移動平均<=75日移動平均")
        if not sma75_up:
            failed_labels.append("75日移動平均が下向き")
        if not sma25_up_3d:
            failed_labels.append("25日移動平均が3営業日連続上向きではない")
        if not drop_ok:
            failed_labels.append(f"高値からの下落率>{core.DROP_MAX:.0f}%")
        if not return_ok:
            failed_labels.append(f"戻り余地<{core.EXPECTED_RISE_MIN:.0f}%")
        if not green_band_ok:
            failed_labels.append(f"押し目安値+{app.CONFIRMED_BAND_PCT:g}%外")
        elif not yellow_band_ok and not (pull_before_latest and rebound_confirmed):
            failed_labels.append(
                f"反発未確認かつ押し目安値+{app.APPROACH_BAND_PCT:g}%外"
            )
        if green_band_ok and not pull_before_latest:
            failed_labels.append("押し目日が最新日")
        if green_band_ok and not rebound_confirmed:
            failed_labels.append("前日終値超えなし")

        row["Conditions_Passed"] = sum(bool(row[column]) for column in CONDITION_COLUMNS)
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
            f"[WARN] 25日線3日上向き判定失敗 {ticker}: {exc}",
            file=sys.stderr,
        )
        return row


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


def save_chart_image_japanese(
    raw_df: pd.DataFrame,
    ticker: str,
    out_dir: str = core.CHART_OUT_DIR,
):
    """ローソク足チャートの軸ラベルを日本語にしてPNG保存する。"""
    if not core.MPF_AVAILABLE:
        return None

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            use = raw_df.loc[:, [(column, ticker) for column in need_cols]].copy()
            use.columns = need_cols
        else:
            use = raw_df[need_cols].copy()
    except Exception:
        return None

    use = use.dropna()
    if use.empty:
        return None

    use = use.tail(core.CHART_LOOKBACK_DAYS)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.png")

    try:
        core.mpf.plot(
            use,
            type="candle",
            mav=(5, 25, 75),
            volume=True,
            style="yahoo",
            ylabel="株価",
            ylabel_lower="出来高",
            savefig=dict(fname=out_path, dpi=140, bbox_inches="tight"),
        )
        return out_path
    except Exception as exc:
        print(
            f"[WARN] 日本語チャート生成失敗 {ticker}: {exc}",
            file=sys.stderr,
        )
        return None


def notify_images_only(df: pd.DataFrame, raw_df: pd.DataFrame):
    """一覧テキストを送らず、候補の画像カードだけをDiscordへ送る。"""
    if df is None or df.empty:
        core.send_long_text(
            f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}\n"
            "該当銘柄はありませんでした。"
        )
        return

    if not core.MPF_AVAILABLE:
        core.send_long_text("チャート生成機能が利用できないため通知を送信できませんでした。")
        return

    ordered = df.sort_values(
        ["Signal_Rank", "Return_%"],
        ascending=[False, False],
    ).head(core.CHART_TOP_N)

    for _, row in ordered.iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        is_green = row.get("Signal_Level") == app.SIGNAL_GREEN
        emoji = "🟢" if is_green else "🟡"
        label = "反発確認" if is_green else "押し目接近"

        rsi = core.latest_rsi_from_raw(raw_df, ticker, period=core.RSI_PERIOD)
        rsi_text = "-" if rsi is None or not np.isfinite(rsi) else f"{rsi:.0f}"

        description = (
            f"📈 戻り余地：{app._fp(row.get('Return_%'), 1)}%\n"
            f"📉 高値からの下落率：{app._fp(row.get('Drop_From_Peak_%'), 1)}%\n"
            f"↩️ 押し目安値からの戻り率："
            f"{app._fp(row.get('Delta_from_Pull_%'), 1)}%\n"
            f"🗓️ 判定期間：{int(row.get('Window') or 0)}日　"
            f"📍 押し目日：{app._fdate(row.get('Pullback_Date'))}\n"
            f"📊 相対力指数（14）：{rsi_text}\n"
            f"💹 最新終値：{app._fp(row.get('Latest_Close'), 0)}\n"
            f"🔻 押し目安値：{app._fp(row.get('Pullback_Low'), 0)}　"
            f"🔺 期間内高値：{app._fp(row.get('Peak_High'), 0)}\n"
            f"📐 25日移動平均：{app._fp(row.get('SMA25'), 0)} "
            "（3営業日連続上向き）\n"
            f"📏 75日移動平均：{app._fp(row.get('SMA75'), 0)}"
        )

        image_path = save_chart_image_japanese(raw_df, ticker)
        if image_path:
            core.discord_send_image_file(
                image_path,
                title=f"{emoji} {label}｜{ticker} {name}".strip(),
                description=description,
            )


def main():
    # main3_precision の保存・選定処理を使い、判定と通知だけ差し替える。
    app.evaluate_one_ticker = evaluate_one_ticker
    app.build_diagnostics = build_diagnostics
    app.notify = notify_images_only
    core.save_chart_image_from_raw = save_chart_image_japanese
    app.main()


if __name__ == "__main__":
    main()
