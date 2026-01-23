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
BREAKING_RSS_QUERY = {
    "label": "【突発】要人発言/領土・安全保障/制裁ショック（6h）",
    "q": "Trump Greenland OR Greenland acquisition OR グリーンランド トランプ OR 米大統領 発言 市場 影響 OR sanctions escalation OR security crisis",
    "lang": "en",
}

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
def fetch_daily(symbol: str, days: int = 60) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
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
      - rss_queries: [{label,q,lang}, ...] (length <= MAX_RSS_QUERIES)
      - rss_urls: Google News RSS URLs
    Policy:
      - Always include one "breaking headline" slot first (Trump/territory/security/sanctions, etc.)
      - Fill the remaining slots with AI-proposed queries (day's market context)
      - Freshness (A): build_google_news_rss_url() always appends when:6h unless query already has when:
    """
    fallback_core = [
        {"label": "中東 原油 供給 リスク（6h）", "q": "中東 原油 供給 リスク 市場 反応", "lang": "ja"},
        {"label": "Taiwan/China military tension（6h）", "q": "Taiwan China military tension market impact", "lang": "en"},
        {"label": "Ukraine/Russia sanctions energy（6h）", "q": "Russia Ukraine sanctions energy prices market impact", "lang": "en"},
    ]

    def finalize(rss_list: List[Dict]) -> Tuple[List[Dict], List[str]]:
        merged: List[Dict] = []
        seen_q = set()

        for it in [BREAKING_RSS_QUERY] + rss_list:
            q = (it.get("q") or "").strip()
            if not q:
                continue
            if q in seen_q:
                continue
            seen_q.add(q)
            merged.append({"label": it.get("label", ""), "q": q, "lang": it.get("lang", "en")})
            if len(merged) >= MAX_RSS_QUERIES:
                break

        urls = [build_google_news_rss_url(x["q"], x.get("lang", "en")) for x in merged]
        return merged, urls

    if not ENABLE_AI:
        return finalize(fallback_core)

    client = openai_client()
    if client is None:
        return finalize(fallback_core)

    vix = safe_float(feat.loc[feat["symbol"] == "VIX", "daily_close"].iloc[0])
    vix15 = safe_float(feat.loc[feat["symbol"] == "VIX", "intraday_%chg_last15m"].iloc[0])
    usdjpy = safe_float(feat.loc[feat["symbol"] == "USDJPY", "daily_close"].iloc[0])
    spx = safe_float(feat.loc[feat["symbol"] == "SPX", "daily_close"].iloc[0])
    nikkei = safe_float(feat.loc[feat["symbol"] == "NIKKEI", "daily_close"].iloc[0])

    system = "You are a markets+geopolitics assistant. Follow the output format strictly."
    user = f"""
次の市場状況に合う「Google News RSS 検索クエリ」を提案して。
日英ミックスOK。地政学とマクロ（米金利/原油/制裁/台湾/中東/要人発言/領土・安全保障など）を広くカバーしつつ、今の数値に寄せて。

Regime: {regime}
Reason: {reason}
VIX: {vix}  VIX15m%: {vix15}
USDJPY: {usdjpy}
SPX: {spx}
NIKKEI: {nikkei}

注意:
- RSS URL生成時に、こちらで自動的に「when:6h」を付与します（あなたは when: を書かなくてOK）。
- ここでは MAX {max(0, MAX_RSS_QUERIES-1)} 件だけ（breaking枠は別途固定）提案してください。

出力は必ずこの形式（各行1件、合計{max(0, MAX_RSS_QUERIES-1)}行）:
lang|label|query

lang は ja または en
label は短い説明（日本語OK）
query は Google News 検索文字列

例:
ja|米金利と株の反応|米金利 上昇 株式 市場 反応
en|Sanctions escalation|sanctions escalation market reaction
""".strip()

    text = call_openai_text(client, system, user)
    if not text:
        return finalize(fallback_core)

    norm: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            continue
        lang, label, q = parts
        lang = lang.lower()
        if lang not in ("ja", "en"):
            continue
        if not label or not q:
            continue
        norm.append({"label": clamp_str(label, 70), "q": q, "lang": lang})
        if len(norm) >= max(0, MAX_RSS_QUERIES - 1):
            break

    if not norm:
        return finalize(fallback_core)

    return finalize(norm)


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
    msgs = ai_build_messages(feat, regime, reason, 
