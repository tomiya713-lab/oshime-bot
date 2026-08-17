# -*- coding: utf-8 -*-
"""
Discord notification wrapper for BBreturn.py.

Purpose:
- Keep the screening logic in BBreturn.py unchanged.
- Replace the long text dump with a compact top-20 summary.
- Send chart notifications for the top 20 candidates.
- Improve readability by grouping price information and indicators.
"""

import math
import pandas as pd

import BBreturn as bot

TOP_N = 20


def fmt_yen(x) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x):,.0f}円"
    except Exception:
        return "-"


def fmt_pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x):+.2f}%"
    except Exception:
        return "-"


def notify(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    ts = bot.now_jst().strftime("%m/%d %H:%M")
    title = "【BB2σタッチ→MA25(+2%以内)】"

    if df is None or df.empty:
        bot.discord_send_text(f"{title} {ts}\n該当なし")
        return

    # Expected upside desc: summary and chart order are identical.
    if "Expected_Up_pct" in df.columns:
        df = df.sort_values("Expected_Up_pct", ascending=False).reset_index(drop=True)

    stock_beta_map, sector_beta_map = bot.load_beta_maps()

    # ---- Compact summary ----
    display_n = min(TOP_N, len(df))
    lines = [
        f"{title} {ts}",
        f"該当: **{len(df)}件**",
        "",
        f"**上位{display_n}件**",
    ]

    for i, r in enumerate(df.head(TOP_N).itertuples(index=False), start=1):
        t = getattr(r, "Ticker")
        name = bot.ticker_name_map.get(t, "")
        up = fmt_pct(getattr(r, "Expected_Up_pct", None))
        lines.append(f"{i}. {t} {name}  **{up}**".rstrip())

    lines.extend(["", f"このあと上位{display_n}件のチャートを送信します"])

    for msg in bot.chunk_text("\n".join(lines)):
        bot.discord_send_text(msg)

    # ---- Chart messages ----
    top = df.head(TOP_N)
    for _, rr in top.iterrows():
        t = rr["Ticker"]
        name = bot.ticker_name_map.get(t, "")

        binfo = stock_beta_map.get(t, {})
        s17 = (binfo.get("S17Nm") or "").strip()
        sinfo = sector_beta_map.get(s17, {})
        sector_beta = sinfo.get("SectorBeta", float("nan"))
        beta = binfo.get("Beta", float("nan"))
        beta_score = binfo.get("BetaScore", float("nan"))

        desc = (
            "**■ 価格情報**\n"
            f"最新: **{fmt_yen(rr.get('Close'))}**\n"
            f"MA25: {fmt_yen(rr.get('SMA25'))}\n"
            f"BB2σタッチ: {fmt_yen(rr.get('Touch_Close'))}\n"
            "\n"
            "**■ 指標**\n"
            f"期待上昇率: **{fmt_pct(rr.get('Expected_Up_pct'))}**\n"
            f"銘柄β: {bot.fp(beta, 3)}\n"
            f"セクターβ: {bot.fp(sector_beta, 3)}\n"
            f"βスコア: {bot.fp(beta_score, 2)}"
        )

        img = bot.save_chart_image_with_bb1sigma(raw_df, t, out_dir=bot.CHART_OUT_DIR)
        if img:
            bot.discord_send_image_file(
                img,
                title=f"{t} {name}".strip(),
                description=desc,
            )


if __name__ == "__main__":
    # Patch only the notification layer; screening/calculation remains in BBreturn.py.
    bot.notify = notify
    bot.main()
