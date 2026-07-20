"""押し目スクリーナーのDiscord本文・画像内指標を日本語化する本番エントリーポイント。"""

import os
import sys
from typing import List

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
            use = raw_df.loc[:, [(col, ticker) for col in need_cols]].copy()
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


def signal_lines(df: pd.DataFrame, level: str, title: str, emoji: str) -> List[str]:
    """スマホで見やすいよう、1銘柄を複数行のブロックで整形する。"""
    subset = df[df["Signal_Level"] == level].copy()
    if subset.empty:
        return [f"{emoji} **{title}：0銘柄**"]

    lines = [f"{emoji} **{title}：{len(subset)}銘柄**", ""]
    for _, row in subset.iterrows():
        ticker = row["Ticker"]
        name = core.TICKER_NAME_MAP.get(ticker, "")
        warning_line = (
            "⚠️ 週足2週移動平均：下向き"
            if bool(row.get("Weekly_MA2_Warning"))
            else ""
        )

        block = [
            f"{emoji} **{ticker} {name}**".rstrip(),
            f"📈 戻り余地：{app._fp(row.get('Return_%'), 1)}%　"
            f"📉 高値下落率：{app._fp(row.get('Drop_From_Peak_%'), 1)}%",
            f"↩️ 押し目戻り率：{app._fp(row.get('Delta_from_Pull_%'), 1)}%",
            f"🗓️ 判定期間：{int(row.get('Window') or 0)}日　"
            f"📍 押し目日：{app._fdate(row.get('Pullback_Date'))}",
        ]
        if warning_line:
            block.append(warning_line)
        block.append("")
        lines.extend(block)

    return lines


def notify_japanese(df: pd.DataFrame, raw_df: pd.DataFrame):
    """黄色・緑の通知本文と画像説明を、スマホ向けに日本語整形する。"""
    condition_text = (
        "📌 **共通条件**\n"
        f"・高値からの下落率 ≤ {core.DROP_MAX:.0f}%\n"
        f"・戻り余地 ≥ {core.EXPECTED_RISE_MIN:.0f}%\n"
        "・25日移動平均 ＞ 75日移動平均\n"
        "・75日移動平均が上向き\n\n"
        "🟡 **押し目接近**\n"
        f"・押し目安値＋{app.APPROACH_BAND_PCT:g}%以内\n"
        "・反発未確認も対象\n\n"
        "🟢 **反発確認**\n"
        f"・押し目安値＋{app.CONFIRMED_BAND_PCT:g}%以内\n"
        "・押し目日は前営業日以前\n"
        "・最新終値が前日終値を上回る\n\n"
        "⚠️ 週足2週移動平均が下向きでも除外せず、警告表示"
    )

    if df is None or df.empty:
        core.send_long_text(
            f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}\n\n"
            f"{condition_text}\n\n"
            "押し目接近・反発確認ともに該当銘柄はありませんでした。\n"
            "診断データ：pullback_diagnostics_latest.csv"
        )
        return

    lines = [
        f"【押し目スクリーニング】{core.now_jst():%m/%d %H:%M}",
        "",
        condition_text,
        "",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    lines.extend(signal_lines(df, app.SIGNAL_GREEN, "反発確認", "🟢"))
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.extend(signal_lines(df, app.SIGNAL_YELLOW, "押し目接近", "🟡"))
    core.send_long_text("\n".join(lines))

    if not core.MPF_AVAILABLE:
        print(
            "[INFO] mplfinance未導入のためチャートを生成しません。",
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
        signal_emoji = "🟢" if row.get("Signal_Level") == app.SIGNAL_GREEN else "🟡"

        rsi = core.latest_rsi_from_raw(raw_df, ticker, period=core.RSI_PERIOD)
        rsi_text = "-" if rsi is None or not np.isfinite(rsi) else f"{rsi:.0f}"
        weekly_note = (
            "\n⚠️ 週足2週移動平均：下向き"
            if bool(row.get("Weekly_MA2_Warning"))
            else ""
        )

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
            f"📐 25日移動平均：{app._fp(row.get('SMA25'), 0)}　"
            f"75日移動平均：{app._fp(row.get('SMA75'), 0)}"
            f"{weekly_note}"
        )

        image_path = save_chart_image_japanese(raw_df, ticker)
        if image_path:
            core.discord_send_image_file(
                image_path,
                title=f"{signal_emoji} {reason.replace(signal_emoji, '').strip()}｜{ticker} {name}".strip(),
                description=description,
            )


def main():
    # 既存の判定・CSV保存ロジックはそのまま使い、表示だけ日本語版へ差し替える。
    app.notify = notify_japanese
    core.save_chart_image_from_raw = save_chart_image_japanese
    app.main()


if __name__ == "__main__":
    main()
