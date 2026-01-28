# -*- coding: utf-8 -*-
"""
Market Regime Monitor + Geopolitics News (Discord) - TEXT MODE (most robust)

Why this version:
- JSON output from the model still broke in your GitHub Actions logs (json.loads Unterminated string...).
- To make it bulletproof, we STOP parsing model JSON and instead ask GPT to output TWO messages in plain text
  with strict delimiters. We then extract the blocks reliably.

Discord outputs:
(1) Regime速報 (short JP)
(2) 地政学ニュース（最大5本、JP要約＋なぜ効く＋URL）＋RSS参照
(3) 表画像（PNG、数値は小数点1桁）

Env:
- DISCORD_WEBHOOK_URL (required)
- OPENAI_API_KEY (optional; if missing -> fallback RSS only)
- OPENAI_MODEL (default: gpt-4.1-mini)
- ENABLE_AI (default: 1)
- AI_ONLY_ON_ALERT (default: 1)  # set 0 for manual-run always AI
- MAX_RSS_QUERIES (default: 3)
- RSS_ITEMS_PER_QUERY (default: 10)
- MAX_NEWS_CANDIDATES (default: 30)
- NEWS_PICK_MAX (default: 5)
- DEBUG_AI (default: 0)  # set 1 to print raw model output in logs (first 1500 chars)
"""

import os
import re
import json
import time
import math
import tempfile
import email.utils
import datetime as dt
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
from openai import OpenAI

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
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "900"))
OPENAI_TIMEOUT_SEC = int(os.environ.get("OPENAI_TIMEOUT_SEC", "30"))
OPENAI_RETRIES = int(os.environ.get("OPENAI_RETRIES", "2"))

MAX_RSS_QUERIES = int(os.environ.get("MAX_RSS_QUERIES", "3"))
RSS_ITEMS_PER_QUERY = int(os.environ.get("RSS_ITEMS_PER_QUERY", "10"))
MAX_NEWS_CANDIDATES = int(os.environ.get("MAX_NEWS_CANDIDATES", "30"))
NEWS_PICK_MAX = int(os.environ.get("NEWS_PICK_MAX", "5"))

# News recency filter (hours)
NEWS_MAX_AGE_HOURS = float(os.environ.get("NEWS_MAX_AGE_HOURS", "6"))
NEWS_FALLBACK_MAX_AGE_HOURS = float(os.environ.get("NEWS_FALLBACK_MAX_AGE_HOURS", "12"))

DEBUG_AI = os.environ.get("DEBUG_AI", "0").strip() == "1"

INTERVAL_INTRADAY = "15m"
LOOKBACK_DAYS = 7
Z_WINDOW = 20

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
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def clamp_str(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"

def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _parse_entry_time(entry) -> Optional[dt.datetime]:
    # Prefer parsed structs (feedparser)
    tt = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if tt:
        try:
            return dt.datetime(*tt[:6])
        except Exception:
            pass
    # Fallback: parse RFC822-like strings if present
    for key in ("published", "updated"):
        s = getattr(entry, key, None)
        if not s:
            continue
        try:
            d = email.utils.parsedate_to_datetime(s)
            if d is None:
                continue
            # Convert to naive UTC datetime
            if d.tzinfo is not None:
                d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
            return d
        except Exception:
            continue
    return None

def _extract_source_from_title(title: str) -> str:
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return ""

def _sanitize_for_prompt(s: str, n: int) -> str:
    """Reduce characters that tend to break structured generations."""
    s = (s or "")
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s).strip()
    return clamp_str(s, n)


# =========================
# yfinance column normalization
# =========================
def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        cols = [c[0] for c in cols]
    norm = []
    for c in cols:
        if isinstance(c, tuple):
            c = c[0] if len(c) > 0 else ""
        norm.append(str(c).strip().lower())
    df.columns = norm
    return df


# =========================
# Market data + features
# =========================
def fetch_daily(symbol: str, days: int = 60) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return _normalize_yf_columns(df)

def fetch_intraday(symbol: str, interval: str = "15m", lookback_days: int = 7) -> pd.DataFrame:
    df = yf.download(symbol, period=f"{lookback_days}d", interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return _normalize_yf_columns(df)

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std

def build_features() -> pd.DataFrame:
    rows = []
    for key, sym in SYMBOLS.items():
        daily = fetch_daily(sym, days=60)
        intra = fetch_intraday(sym, interval=INTERVAL_INTRADAY, lookback_days=LOOKBACK_DAYS)

        if daily.empty or "close" not in daily.columns:
            rows.append({
                "symbol": key,
                "daily_close": None,
                "daily_%chg_1d": None,
                "daily_%chg_5d": None,
                "intraday_close_15m": None,
                "intraday_%chg_last15m": None,
                "zscore_20d": None,
            })
            continue

        close = daily["close"].iloc[-1]
        chg1 = (daily["close"].pct_change(1).iloc[-1] * 100.0) if len(daily) >= 2 else None
        chg5 = (daily["close"].pct_change(5).iloc[-1] * 100.0) if len(daily) >= 6 else None
        z20 = zscore(daily["close"], Z_WINDOW).iloc[-1] if len(daily) >= Z_WINDOW else None

        intra_close, intra_chg = None, None
        if not intra.empty and "close" in intra.columns and len(intra) >= 2:
            intra_close = intra["close"].iloc[-1]
            intra_chg = intra["close"].pct_change(1).iloc[-1] * 100.0

        rows.append({
            "symbol": key,
            "daily_close": float(close) if pd.notna(close) else None,
            "daily_%chg_1d": float(chg1) if chg1 is not None and pd.notna(chg1) else None,
            "daily_%chg_5d": float(chg5) if chg5 is not None and pd.notna(chg5) else None,
            "intraday_close_15m": float(intra_close) if intra_close is not None and pd.notna(intra_close) else None,
            "intraday_%chg_last15m": float(intra_chg) if intra_chg is not None and pd.notna(intra_chg) else None,
            "zscore_20d": float(z20) if z20 is not None and pd.notna(z20) else None,
        })
    return pd.DataFrame(rows)

def eval_regime(feat: pd.DataFrame) -> Tuple[str, str]:
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
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================
# Google News RSS fetch
# =========================
def build_google_news_rss_url(query: str, lang: str) -> str:
    if lang == "ja":
        hl, gl, ceid = "ja", "JP", "JP:ja"
    else:
        hl, gl, ceid = "en-US", "US", "US:en"
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

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
                "snippet": _sanitize_for_prompt(summary, 200),
                "published_utc": published.isoformat() if published else None,
                "source": _extract_source_from_title(title),
            })
        return items
    except Exception as e:
        print(f"[WARN] RSS fetch failed: {url} ({e})")
        return []

def _dt_from_iso(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # Accept "2026-01-19T12:34:56" or "...Z"
        s2 = s.replace("Z", "")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None

def _filter_by_age(items: List[Dict], max_age_hours: float, keep_undated: bool = False) -> List[Dict]:
    """Keep items within max_age_hours from now (UTC). If keep_undated=True, include items with no timestamp."""
    now_utc = dt.datetime.utcnow()
    out = []
    for it in items:
        t = _dt_from_iso((it.get('published_utc') or '').strip())
        if t is None:
            if keep_undated:
                out.append(it)
            continue
        age_h = (now_utc - t).total_seconds() / 3600.0
        if age_h <= max_age_hours:
            out.append(it)
    return out

def collect_news_candidates(rss_urls: List[str]) -> List[Dict]:
    all_items: List[Dict] = []
    for url in rss_urls:
        all_items.extend(fetch_rss_items(url, RSS_ITEMS_PER_QUERY))
        time.sleep(0.2)

    # Deduplicate by URL first
    uniq: Dict[str, Dict] = {}
    for it in all_items:
        u = it.get("url", "")
        if not u:
            continue
        if u not in uniq:
            uniq[u] = it

    items = list(uniq.values())
    items.sort(key=lambda x: x.get("published_utc") or "", reverse=True)

    # Hard recency filter (machine)
    fresh = _filter_by_age(items, NEWS_MAX_AGE_HOURS, keep_undated=False)
    if len(fresh) >= min(5, MAX_NEWS_CANDIDATES):
        items_use = fresh
        age_used = NEWS_MAX_AGE_HOURS
    else:
        # Fallback: allow a bit older if news volume is low
        items_use = _filter_by_age(items, NEWS_FALLBACK_MAX_AGE_HOURS, keep_undated=True)
        age_used = NEWS_FALLBACK_MAX_AGE_HOURS

    items_use = items_use[:MAX_NEWS_CANDIDATES]

    if DEBUG_AI:
        print(f"[DEBUG] news candidates: total={len(items)} within{NEWS_MAX_AGE_HOURS}h={len(fresh)} using<= {age_used}h -> {len(items_use)}")
        if items_use:
            print("[DEBUG] newest candidate published_utc:", items_use[0].get("published_utc"))

    return items_use


# =========================
# OpenAI helpers (TEXT with delimiters)
# =========================
def openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY is empty")
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def call_openai_text(client: OpenAI, system: str, user: str) -> Optional[str]:
    for attempt in range(1 + max(0, OPENAI_RETRIES)):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=OPENAI_MAX_TOKENS,
                timeout=OPENAI_TIMEOUT_SEC,
            )
            text = (resp.choices[0].message.content or "").strip()
            if DEBUG_AI:
                print("[DEBUG] raw model output (head):")
                print(text[:1500])
            if text:
                return text
        except Exception as e:
            print(f"[WARN] OpenAI text failed (attempt {attempt+1}): {e}")
            time.sleep(0.6)
    return None

def extract_block(text: str, start: str, end: str) -> Optional[str]:
    m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def ai_propose_rss_queries(feat: pd.DataFrame, regime: str, reason: str) -> Tuple[List[Dict], List[str]]:
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

    vix = safe_float(feat.loc[feat["symbol"] == "VIX", "daily_close"].iloc[0])
    vix15 = safe_float(feat.loc[feat["symbol"] == "VIX", "intraday_%chg_last15m"].iloc[0])
    usdjpy = safe_float(feat.loc[feat["symbol"] == "USDJPY", "daily_close"].iloc[0])

    system = "You are a markets+geopolitics assistant. Follow the output format strictly."
    user = f"""
次の市場状況に合う「Google News RSS 検索クエリ」を最大{MAX_RSS_QUERIES}本提案して。
日英ミックスOK。地政学とマクロ（米金利/原油/制裁/台湾/中東など）を広くカバーしつつ、今の数値に寄せて。

Regime: {regime}
Reason: {reason}
VIX: {vix}  VIX15m%: {vix15}  USDJPY: {usdjpy}

出力は必ずこの形式（各行1件、合計{MAX_RSS_QUERIES}行）:
lang|label|query

lang は ja または en
label は短い説明（日本語OK）
query は Google News 検索文字列

例:
ja|中東原油供給リスク|中東 原油 供給 リスク
en|Taiwan tension|Taiwan China military tension
""".strip()

    text = call_openai_text(client, system, user)
    if not text:
        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in fallback[:MAX_RSS_QUERIES]]
        return fallback[:MAX_RSS_QUERIES], urls

    norm = []
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
        if len(norm) >= MAX_RSS_QUERIES:
            break

    if not norm:
        urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in fallback[:MAX_RSS_QUERIES]]
        return fallback[:MAX_RSS_QUERIES], urls

    urls = [build_google_news_rss_url(x["q"], x["lang"]) for x in norm]
    return norm, urls


def ai_build_messages(
    feat: pd.DataFrame,
    regime: str,
    reason: str,
    rss_queries: List[Dict],
    news_candidates: List[Dict],
) -> Optional[Tuple[str, str]]:
    if not ENABLE_AI:
        return None

    client = openai_client()
    if client is None:
        return None

    if AI_ONLY_ON_ALERT and regime == "NORMAL":
        return None

    def pick(sym: str) -> dict:
        row = feat.loc[feat["symbol"] == sym].iloc[0].to_dict()
        return {
            "close": safe_float(row.get("daily_close")),
            "chg1": safe_float(row.get("daily_%chg_1d")),
            "chg5": safe_float(row.get("daily_%chg_5d")),
            "chg15": safe_float(row.get("intraday_%chg_last15m")),
            "z20": safe_float(row.get("zscore_20d")),
        }

    market = {"VIX": pick("VIX"), "USDJPY": pick("USDJPY"), "NIKKEI": pick("NIKKEI"), "SPX": pick("SPX")}

    # Keep prompt compact and safe
    cand_lines = []
    for it in news_candidates[:MAX_NEWS_CANDIDATES]:
        title = _sanitize_for_prompt(it.get("title", ""), 160)
        source = _sanitize_for_prompt(it.get("source", ""), 60)
        url = (it.get("url", "") or "").strip()
        snippet = _sanitize_for_prompt(it.get("snippet", ""), 200)
        published = (it.get("published_utc", "") or "")
        if title and url:
            cand_lines.append(f"- {title} | {source} | {published} | {snippet} | {url}")
    cand_blob = "\n".join(cand_lines)

    system = "You are an editor who writes concise, actionable Discord messages in Japanese."
    user = f"""
あなたは「市場データを解釈し、関連する地政学ニュースを選別して要約する」編集者です。
以下を読み、Discord用に2つのメッセージを作成してください。出力は必ず区切り付きの“生テキスト”のみ。

【市場データ】
ts_jst: {fmt_ts_jst()}
regime: {regime}
reason: {reason}
market: {json.dumps(market, ensure_ascii=False)}

【要件】
(1) 1通目：結論＋解釈（短い）
- Regime（NORMAL/ALERT/CRISIS）
- なぜそう見えるか（VIX/円/先物/米株の変化を日本語で1〜3行）
- 監視ポイント（次の1〜2時間で見るべき指標）

(2) 2通目：地政学ニュース（最大{NEWS_PICK_MAX}本）
- 候補（最大{MAX_NEWS_CANDIDATES}件）から重要度が高いものだけ選ぶ
- 各ニュース: 見出し（ソース）＋一言要約（日本語）＋「なぜ効く可能性があるか」＋URL
- 直近優先、重複回避
- 最後にRSS候補（参照）も3本つける（label + URL）

【出力フォーマット厳守】
<<<MSG1>>>
...ここに1通目...
<<<ENDMSG1>>>
<<<MSG2>>>
...ここに2通目...
<<<ENDMSG2>>>

【RSS候補】
{json.dumps(rss_queries, ensure_ascii=False)}

【ニュース候補一覧】
{cand_blob}

【重要】原則として直近6時間以内のニュースを最優先で選ぶこと。6時間以内が少ない場合のみ、12時間以内まで許容。
""".strip()

    text = call_openai_text(client, system, user)
    if not text:
        return None

    msg1 = extract_block(text, "<<<MSG1>>>", "<<<ENDMSG1>>>")
    msg2 = extract_block(text, "<<<MSG2>>>", "<<<ENDMSG2>>>")
    if not msg1 or not msg2:
        return None
    return msg1, msg2


# =========================
# Fallback messages
# =========================
def build_fallback_messages(regime: str, reason: str, rss_queries: List[Dict]) -> Tuple[str, str]:
    ts = fmt_ts_jst()
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

    rss_queries, rss_urls = ai_propose_rss_queries(feat, regime, reason)
    candidates = collect_news_candidates(rss_urls)
    msgs = ai_build_messages(feat, regime, reason, rss_queries, candidates)

    if not msgs:
        msg1, msg2 = build_fallback_messages(regime, reason, rss_queries)
    else:
        msg1, msg2 = msgs

    discord_send_text(DISCORD_WEBHOOK_URL, msg1)
    discord_send_text(DISCORD_WEBHOOK_URL, msg2)

    title = f"【Market Regime Monitor】{ts}  Regime={regime}\nReason: {reason}"
    with tempfile.TemporaryDirectory() as d:
        png_path = os.path.join(d, "regime_table.png")
        render_table_png(feat, title=title, out_path=png_path)
        discord_send_file(DISCORD_WEBHOOK_URL, title, png_path)

def main():
    if not DISCORD_WEBHOOK_URL:
        print("[ERROR] DISCORD_WEBHOOK_URL is not set")
        return
    feat = build_features()
    notify(feat)

if __name__ == "__main__":
    main()
