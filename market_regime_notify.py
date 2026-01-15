# -*- coding: utf-8 -*-
import os
import re
import json
import time
import math
import tempfile
import datetime as dt
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import yfinance as yf

import feedparser  # pip install feedparser
from openai import OpenAI  # pip install openai

# matplotlib は mplfinance 依存で入っていることが多い
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Config (Env)
# =========================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

ENABLE_AI = os.environ.get("ENABLE_AI", "1").strip() == "1"
AI_ONLY_ON_ALERT = os.environ.get("AI_ONLY_ON_ALERT", "1").strip() == "1"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "700"))
OPENAI_TIMEOUT_SEC = int(os.environ.get("OPENAI_TIMEOUT_SEC", "30"))

# RSS settings
RSS_LANG_PRIMARY = os.environ.get("RSS_LANG_PRIMARY", "ja").strip()  # "ja"
RSS_LANG_SECONDARY = os.environ.get("RSS_LANG_SECONDARY", "en").strip()  # "en"
MAX_RSS_QUERIES = int(os.environ.get("MAX_RSS_QUERIES", "3"))  # AI提案URLの最大本数
RSS_ITEMS_PER_QUERY = int(os.environ.get("RSS_ITEMS_PER_QUERY", "10"))  # 1URLあたり上限
MAX_NEWS_CANDIDATES = int(os.environ.get("MAX_NEWS_CANDIDATES", "30"))  # 候補の上限
NEWS_PICK_MAX = int(os.environ.get("NEWS_PICK_MAX", "5"))  # 2通目の最大本数

# Market data configuration
INTERVAL_INTRADAY = "15m"
LOOKBACK_DAYS = 7  # 15m 取得期間の目安
Z_WINDOW = 20  # zscore window

# Symbols
SYMBOLS = {
    "VIX": "^VIX",
    "USDJPY": "JPY=X",
    "NIKKEI": "^N225",
    "NIKKEI_FUT": "NK=F",
    "SPX": "^GSPC",
}


# =========================
# Utils
# =========================
def jst_now() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=9)

def fmt_ts_jst(ts: Optional[dt.datetime] = None) -> str:
    ts = ts or jst_now()
    return ts.strftime("%m/%d %H:%M")

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def clamp_str(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _parse_entry_time(entry) -> Optional[dt.datetime]:
    # feedparser gives published_parsed / updated_parsed as time.struct_time
    tt = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if tt:
        try:
            return dt.datetime(*tt[:6])
        except Exception:
            return None
    return None


# =========================
# Market data + features
# =========================

def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance が MultiIndex columns を返すケースに対応
    cols = df.columns
    # MultiIndex or tuple columns
    if isinstance(cols, pd.MultiIndex):
        cols = [c[0] for c in cols]  # 例: ('Close', '^VIX') -> 'Close'
    norm = []
    for c in cols:
        if isinstance(c, tuple):
            c = c[0] if len(c) > 0 else ""
        c = str(c).strip().lower()
        norm.append(c)
    df.columns = norm
    return df

def fetch_daily(symbol: str, days: int = 60) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _normalize_yf_columns(df)
    df.index = pd.to_datetime(df.index)
    return df

def fetch_intraday(symbol: str, interval: str = "15m", lookback_days: int = 7) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{lookback_days}d", interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index)
    return df

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std

def build_features() -> pd.DataFrame:
    rows = []
    for key, sym in SYMBOLS.items():
        daily = fetch_daily(sym, days=60)
        intra = fetch_intraday(sym, interval=INTERVAL_INTRADAY, lookback_days=LOOKBACK_DAYS)

        if daily.empty:
            rows.append({
                "symbol": key,
                "daily_close": None,
                "daily_%chg_1d": None,
                "daily_%chg_5d": None,
                "intraday_close_15m": None,
                "intraday_%chg_last15m": None,
                "zscore_20d": None
            })
            continue

        close = daily["close"].iloc[-1]
        chg1 = (daily["close"].pct_change(1).iloc[-1] * 100.0) if len(daily) >= 2 else None
        chg5 = (daily["close"].pct_change(5).iloc[-1] * 100.0) if len(daily) >= 6 else None
        z20 = zscore(daily["close"], Z_WINDOW).iloc[-1] if len(daily) >= Z_WINDOW else None

        intra_close = None
        intra_chg = None
        if not intra.empty and len(intra) >= 2:
            intra_close = intra["close"].iloc[-1]
            intra_chg = (intra["close"].pct_change(1).iloc[-1] * 100.0)

        rows.append({
            "symbol": key,
            "daily_close": float(close) if pd.notna(close) else None,
            "daily_%chg_1d": float(chg1) if chg1 is not None and pd.notna(chg1) else None,
            "daily_%chg_5d": float(chg5) if chg5 is not None and pd.notna(chg5) else None,
            "intraday_close_15m": float(intra_close) if intra_close is not None and pd.notna(intra_close) else None,
            "intraday_%chg_last15m": float(intra_chg) if intra_chg is not None and pd.notna(intra_chg) else None,
            "zscore_20d": float(z20) if z20 is not None and pd.notna(z20) else None
        })

    return pd.DataFrame(rows)

def eval_regime(feat: pd.DataFrame) -> Tuple[str, str]:
    """
    Simple regime:
      - CRISIS: VIX >= 25 or VIX 15m% >= 10
      - ALERT : VIX >= 18 or VIX 15m% >= 6
      - NORMAL: else
    """
    vix_row = feat.loc[feat["symbol"] == "VIX"].iloc[0].to_dict()
    vix = safe_float(vix_row.get("daily_close"))
    vix15 = safe_float(vix_row.get("intraday_%chg_last15m"))

    if vix is None:
        return "NORMAL", "VIX data missing"

    reason = f"VIX={vix:.2f}"
    if vix15 is not None:
        reason += f" | VIX15m{vix15:+.2f}%"

    if (vix >= 25) or (vix15 is not None and vix15 >= 10):
        return "CRISIS", reason
    if (vix >= 18) or (vix15 is not None and vix15 >= 6):
        return "ALERT", reason
    return "NORMAL", reason


# =========================
# Discord senders
# =========================
def discord_send_text(webhook_url: str, content: str) -> None:
    if not webhook_url:
        print("[WARN] DISCORD_WEBHOOK_URL is empty")
        return
    r = requests.post(webhook_url, json={"content": content}, timeout=20)
    if r.status_code >= 300:
        print(f"[WARN] Discord send text failed: {r.status_code} {r.text[:200]}")

def discord_send_file(webhook_url: str, content: str, filepath: str) -> None:
    if not webhook_url:
        print("[WARN] DISCORD_WEBHOOK_URL is empty")
        return
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        payload = {"content": content}
        data = {"payload_json": json.dumps(payload, ensure_ascii=False)}
        r = requests.post(webhook_url, data=data, files=files, timeout=30)
    if r.status_code >= 300:
        print(f"[WARN] Discord send file failed: {r.status_code} {r.text[:200]}")


# =========================
# Table image
# =========================
def render_table_png(feat: pd.DataFrame, title: str, out_path: str) -> None:
    df = feat.copy()

    # display formatting
    # 小数点第1位、欠損は "-"
    def fmt_val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        try:
            return f"{float(v):.1f}"
        except Exception:
            return "-"

    display_cols = ["symbol", "daily_close", "daily_%chg_1d", "daily_%chg_5d",
                    "intraday_close_15m", "intraday_%chg_last15m", "zscore_20d"]
    df = df[display_cols]
    df = df.rename(columns={
        "symbol": "symbol",
        "daily_close": "daily_close",
        "daily_%chg_1d": "1d%",
        "daily_%chg_5d": "5d%",
        "intraday_close_15m": "intra_close",
        "intraday_%chg_last15m": "15m%",
        "zscore_20d": "z20",
    })

    for c in df.columns:
        if c != "symbol":
            df[c] = df[c].apply(fmt_val)

    fig_w = 10
    fig_h = 2.0 + 0.35 * (len(df) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    # header bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================
# Google News RSS fetch
# =========================
def fetch_rss_items(url: str, per_query: int) -> List[Dict]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        items = []
        for entry in feed.entries[:per_query]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = _clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            published = _parse_entry_time(entry)
            items.append({
                "title": title,
                "url": link,
                "snippet": clamp_str(summary, 220),
                "published_utc": published.isoformat() if published else None,
                "source": _extract_source_from_title(title),
            })
        return items
    except Exception as e:
        print(f"[WARN] RSS fetch failed: {url} ({e})")
        return []

def _extract_source_from_title(title: str) -> str:
    # Google News RSS title is often "Headline - Publisher"
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        return parts[-1].strip()
    return ""


# =========================
# OpenAI helpers
# =========================
def openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY is empty")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def _safe_json_extract(text: str) -> Optional[dict]:
    if not text:
        return None
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def ai_propose_rss_queries(feat: pd.DataFrame, regime: str, reason: str) -> Tuple[List[Dict], List[str]]:
    """
    Returns:
      - queries: list of dict {label, q, lang}
      - rss_urls: built urls (max MAX_RSS_QUERIES)
    """
    # Default fallback (日英ミックス)
    fallback = [
        {"label": "中東 原油 供給 リスク - risk-off時の定番", "q": "中東 原油 供給 リスク", "lang": "ja"},
        {"label": "Taiwan China military tension - 地政学リスク", "q": "Taiwan China military tension", "lang": "en"},
        {"label": "Russia Ukraine sanctions energy prices - 制裁/エネルギー", "q": "Russia Ukraine sanctions energy prices", "lang": "en"},
    ]

    if not ENABLE_AI:
        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in fallback[:MAX_RSS_QUERIES]]
        return fallback[:MAX_RSS_QUERIES], urls

    client = openai_client()
    if client is None:
        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in fallback[:MAX_RSS_QUERIES]]
        return fallback[:MAX_RSS_QUERIES], urls

    # Summarize key numbers for the prompt
    vix = safe_float(feat.loc[feat["symbol"] == "VIX", "daily_close"].iloc[0])
    vix15 = safe_float(feat.loc[feat["symbol"] == "VIX", "intraday_%chg_last15m"].iloc[0])
    usdjpy = safe_float(feat.loc[feat["symbol"] == "USDJPY", "daily_close"].iloc[0])
    nikkei = safe_float(feat.loc[feat["symbol"] == "NIKKEI", "daily_close"].iloc[0])
    spx = safe_float(feat.loc[feat["symbol"] == "SPX", "daily_close"].iloc[0])

    prompt = f"""
あなたは市場の地政学リスク監視アシスタントです。
以下の市場データから「今このタイミングで関連しやすい地政学/マクロのニュース探索クエリ」を提案してください。
日英ミックスで最大{MAX_RSS_QUERIES}本。
出力はJSONのみ。

【市場状態】
Regime: {regime}
Reason: {reason}

【主要指標（参考）】
VIX: {vix}
VIX 15m %: {vix15}
USDJPY: {usdjpy}
NIKKEI: {nikkei}
SPX: {spx}

【要件】
- 各クエリは Google News RSS search の q= にそのまま使える文字列
- lang は "ja" または "en"
- label はDiscord表示用の短い説明（30文字程度）
- クエリは地政学（中東/紅海/ウクライナ/台湾/制裁/原油/海運/テロ等）と、状況に応じて金利や為替なども混ぜてOK
- 重複しないように

JSON形式:
{{
  "queries": [
    {{"label":"...", "q":"...", "lang":"ja"}},
    ...
  ]
}}
""".strip()

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            timeout=OPENAI_TIMEOUT_SEC,
        )
        text = (resp.output_text or "").strip()
        data = _safe_json_extract(text)
        if not data or "queries" not in data:
            raise ValueError("No queries JSON")

        queries = data["queries"][:MAX_RSS_QUERIES]
        # validate
        norm = []
        for q in queries:
            label = str(q.get("label", "")).strip()
            qq = str(q.get("q", "")).strip()
            lang = str(q.get("lang", "")).strip().lower()
            if not label or not qq or lang not in ("ja", "en"):
                continue
            norm.append({"label": clamp_str(label, 60), "q": qq, "lang": lang})
        if not norm:
            raise ValueError("All queries invalid")

        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in norm]
        return norm, urls
    except Exception as e:
        print(f"[WARN] OpenAI propose queries failed: {e}")
        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in fallback[:MAX_RSS_QUERIES]]
        return fallback[:MAX_RSS_QUERIES], urls

def build_google_news_rss_url(query: str, lang: str) -> str:
    # lang-> hl/gl/ceid
    if lang == "ja":
        hl, gl, ceid = "ja", "JP", "JP:ja"
    else:
        hl, gl, ceid = "en-US", "US", "US:en"
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def collect_news_candidates(rss_urls: List[str]) -> List[Dict]:
    all_items = []
    for url in rss_urls:
        items = fetch_rss_items(url, RSS_ITEMS_PER_QUERY)
        all_items.extend(items)
        time.sleep(0.2)

    # Dedupe by URL
    uniq = {}
    for it in all_items:
        u = it.get("url", "")
        if not u:
            continue
        if u not in uniq:
            uniq[u] = it
        else:
            # keep newer if available
            old_t = uniq[u].get("published_utc")
            new_t = it.get("published_utc")
            if new_t and (not old_t or new_t > old_t):
                uniq[u] = it

    items = list(uniq.values())

    # Sort by published desc (fallback: keep order)
    def sort_key(x):
        return x.get("published_utc") or ""
    items.sort(key=sort_key, reverse=True)

    return items[:MAX_NEWS_CANDIDATES]

def ai_build_messages(
    feat: pd.DataFrame,
    regime: str,
    reason: str,
    rss_queries: List[Dict],
    news_candidates: List[Dict],
) -> Optional[Tuple[str, str]]:
    """
    Returns (msg1, msg2) or None on failure.
    msg1: conclusion + interpretation (short)
    msg2: top geopolitics news (<= NEWS_PICK_MAX)
    """
    if not ENABLE_AI:
        return None

    client = openai_client()
    if client is None:
        return None

    # If AI_ONLY_ON_ALERT is enabled, skip AI on NORMAL
    if AI_ONLY_ON_ALERT and regime == "NORMAL":
        return None

    # Compact features for interpretation
    def pick(sym: str) -> dict:
        row = feat.loc[feat["symbol"] == sym].iloc[0].to_dict()
        return {
            "close": safe_float(row.get("daily_close")),
            "chg1": safe_float(row.get("daily_%chg_1d")),
            "chg5": safe_float(row.get("daily_%chg_5d")),
            "chg15": safe_float(row.get("intraday_%chg_last15m")),
            "z20": safe_float(row.get("zscore_20d")),
        }

    payload = {
        "ts_jst": fmt_ts_jst(),
        "regime": regime,
        "reason": reason,
        "market": {
            "VIX": pick("VIX"),
            "USDJPY": pick("USDJPY"),
            "NIKKEI": pick("NIKKEI"),
            "SPX": pick("SPX"),
        },
        "rss_queries": rss_queries,
        "news_candidates": news_candidates,  # up to 30
        "constraints": {
            "news_pick_max": NEWS_PICK_MAX,
            "language": "ja",
            "style": "discord_short",
        }
    }

    prompt = f"""
あなたは「市場データを解釈し、関連する地政学ニュースを選別して要約する」編集者です。
以下のJSON入力を読み、Discordに投稿する2つのメッセージを作ってください（日本語）。
- 1通目: 結論＋解釈（短い）
  - Regime（NORMAL/ALERT/CRISIS）
  - なぜそう見えるか（1〜3行）
  - 監視ポイント（次の1〜2時間で見るべき指標）
- 2通目: 地政学ニュース（最大{NEWS_PICK_MAX}本）
  - 見出し（ソース）＋一言要約（日本語）
  - “効く可能性がある理由”を一言
  - URL（そのまま貼る）
  - なるべく直近のニュースを優先。重複は避ける。
  - 候補が弱い/不足なら「該当強いニュースなし」を明記。

【重要】
- news_candidates は「タイトル/時刻/短い抜粋」だけです。本文の断定は避け、推測は推測と分かる表現にしてください。
- 出力はJSONのみ。

出力JSON形式:
{{
  "msg1": "...",
  "news": [
    {{"title":"...", "source":"...", "summary_ja":"...", "why":"...", "url":"..."}}
  ],
  "notes": "任意（短く）"
}}

入力JSON:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            timeout=OPENAI_TIMEOUT_SEC,
        )
        text = (resp.output_text or "").strip()
        data = _safe_json_extract(text)
        if not data or "msg1" not in data:
            raise ValueError("Invalid JSON from model")

        msg1 = str(data.get("msg1", "")).strip()
        news = data.get("news", []) or []

        # Build msg2
        lines = ["【地政学ニュース】"]
        picked = 0
        for n in news:
            if picked >= NEWS_PICK_MAX:
                break
            title = clamp_str(str(n.get("title", "")).strip(), 120)
            source = clamp_str(str(n.get("source", "")).strip(), 40)
            summary = clamp_str(str(n.get("summary_ja", "")).strip(), 120)
            why = clamp_str(str(n.get("why", "")).strip(), 90)
            url = str(n.get("url", "")).strip()

            if not title or not url:
                continue

            head = f"- {title}"
            if source:
                head += f"（{source}）"
            lines.append(head)
            if summary:
                lines.append(f"  要約: {summary}")
            if why:
                lines.append(f"  なぜ効く: {why}")
            lines.append(f"  {url}")
            picked += 1

        if picked == 0:
            lines.append("- 該当の強い地政学ニュースが候補から見つかりませんでした。")
            # 透明性：RSS候補URLを残す（後追い検証用）
        lines.append("")
        lines.append("【RSS候補（参照）】")
        for q in rss_queries:
            label = q.get("label", "")
            url = build_google_news_rss_url(q.get("q", ""), q.get("lang", "ja"))
            lines.append(f"- {label} {url}")

        msg2 = "\n".join(lines).strip()

        if not msg1:
            raise ValueError("Empty msg1")

        return msg1, msg2
    except Exception as e:
        print(f"[WARN] OpenAI build messages failed: {e}")
        return None


# =========================
# Fallback messages (no AI)
# =========================
def build_fallback_messages(regime: str, reason: str, rss_queries: List[Dict]) -> Tuple[str, str]:
    ts = fmt_ts_jst()
    vix = safe_float(reason.split("VIX=")[-1].split("|")[0]) if "VIX=" in reason else None
    msg1 = f"【Regime速報】{ts} Regime={regime}\n{reason}\n監視: VIX/ドル円/日経先物の次の1〜2本（15m）"
    lines = ["【地政学ニュース探索】RSS候補（クリック可）"]
    for q in rss_queries:
        label = q.get("label", "")
        url = build_google_news_rss_url(q.get("q", ""), q.get("lang", "ja"))
        lines.append(f"{label}{url}")
    msg2 = "\n".join(lines)
    return msg1, msg2


# =========================
# Main notify
# =========================
def notify(feat: pd.DataFrame) -> None:
    ts = fmt_ts_jst()
    regime, reason = eval_regime(feat)

    # 1) AI proposes RSS queries (or fallback)
    rss_queries, rss_urls = ai_propose_rss_queries(feat, regime, reason)

    # 2) Collect RSS candidates (max 30)
    candidates = collect_news_candidates(rss_urls)

    # 3) AI builds msg1+msg2 (or fallback)
    msgs = ai_build_messages(feat, regime, reason, rss_queries, candidates)

    if not msgs:
        msg1, msg2 = build_fallback_messages(regime, reason, rss_queries)
    else:
        msg1, msg2 = msgs

    # Send (1) and (2) text
    discord_send_text(DISCORD_WEBHOOK_URL, msg1)
    discord_send_text(DISCORD_WEBHOOK_URL, msg2)

    # (3) Table as image + keep short title text
    title = f"【Market Regime Monitor】{ts}  Regime={regime}\nReason: {reason}"
    with tempfile.TemporaryDirectory() as d:
        png_path = os.path.join(d, "regime_table.png")
        render_table_png(feat, title="Market Regime Table (rounded to 0.1)", out_path=png_path)
        discord_send_file(DISCORD_WEBHOOK_URL, title, png_path)


def main():
    if not DISCORD_WEBHOOK_URL:
        print("[ERROR] DISCORD_WEBHOOK_URL is not set")
        return
    feat = build_features()
    notify(feat)

if __name__ == "__main__":
    main()
