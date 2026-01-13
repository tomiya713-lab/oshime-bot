import os
import sys
import json
import re
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ====== 基本設定（既存運用に寄せる） ======
TZ_OFFSET = 9  # JST
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# --- AI (optional) ---
ENABLE_AI = os.getenv("ENABLE_AI", "1") == "1"
AI_ONLY_ON_ALERT = os.getenv("AI_ONLY_ON_ALERT", "1") == "1"  # NORMAL時はAI処理を省いて節約
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # 安くするなら gpt-4.1-nano 等も検討
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "30"))

# Google News RSS
RSS_LANG_PRIMARY = os.getenv("RSS_LANG_PRIMARY", "ja")   # ja / en
RSS_LANG_SECONDARY = os.getenv("RSS_LANG_SECONDARY", "en")
MAX_RSS_QUERIES = int(os.getenv("MAX_RSS_QUERIES", "3"))  # AIが提案する検索クエリ数
RSS_ITEMS_PER_QUERY = int(os.getenv("RSS_ITEMS_PER_QUERY", "8"))
NEWS_PICK_MAX = int(os.getenv("NEWS_PICK_MAX", "5"))

# 取得範囲（この値は運用しながら調整OK）
DAILY_PERIOD = os.getenv("DAILY_PERIOD", "120d")       # 日足の取得期間
INTRADAY_PERIOD = os.getenv("INTRADAY_PERIOD", "14d")  # 15分足の取得期間
INTRADAY_INTERVAL = os.getenv("INTRADAY_INTERVAL", "15m")

# レジーム判定（簡易）
VIX_LEVEL_ALERT = float(os.getenv("VIX_LEVEL_ALERT", "20"))
VIX_LEVEL_CRISIS = float(os.getenv("VIX_LEVEL_CRISIS", "25"))
VIX_SPIKE_15M_PCT = float(os.getenv("VIX_SPIKE_15M_PCT", "3.0"))
USDJPY_DROP_1D_PCT = float(os.getenv("USDJPY_DROP_1D_PCT", "-0.5"))       # 円高方向（USDJPY下落）
NIKKEI_FUT_DROP_1D_PCT = float(os.getenv("NIKKEI_FUT_DROP_1D_PCT", "-0.7"))

TICKERS = {
    "VIX": "^VIX",
    "USDJPY": "JPY=X",
    "NIKKEI": "^N225",
    "NIKKEI_FUT": "NK=F",
    "SPX": "^GSPC",
}

# ===== ユーティリティ（既存運用のまま寄せる） =====
def now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def chunk_text(text: str, limit: int = 1900):
    out, buf, size = [], [], 0
    for line in text.splitlines():
        add = len(line) + 1
        if size + add > limit and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += add
    if buf:
        out.append("\n".join(buf))
    return out

def discord_post(payload: dict):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL is empty. skip notify.", file=sys.stderr)
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=30)
        if r.status_code >= 300:
            print(f"[WARN] Discord post failed: {r.status_code} {r.text}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[WARN] Discord post error: {e}", file=sys.stderr)
        return False

def discord_send_text(content: str):
    return discord_post({"content": content})

def fp(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

# ===== データ取得（MultiIndex/tuple列も落ちないように） =====
def _flatten_columns(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if x not in (None, "", " ")]
            out.append(parts[0] if parts else "")
        else:
            out.append(str(c))
    return out

def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # indexは基本UTC想定 → JST相当で扱えるようにする（tz有無両対応）
    idx = df.index
    if getattr(idx, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Tokyo")

    # 列名が tuple / MultiIndex の場合に備える
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = _flatten_columns(df.columns)

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def last_valid_close(df: pd.DataFrame) -> float:
    if df is None or df.empty or "close" not in df:
        return np.nan
    s = df["close"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def pct_change(a, b):
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0

def zscore_last(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < max(5, window):
        return np.nan
    w = s.iloc[-window:]
    mu = w.mean()
    sd = w.std(ddof=0)
    if sd == 0:
        return np.nan
    return float((w.iloc[-1] - mu) / sd)

# ===== 特徴量作成（あなたのCSVと同列） =====
def build_features() -> pd.DataFrame:
    rows = []
    for name, sym in TICKERS.items():
        d = fetch_ohlc(sym, period=DAILY_PERIOD, interval="1d")
        i = fetch_ohlc(sym, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)

        d_close = last_valid_close(d)
        i_close = last_valid_close(i)

        # daily changes
        d_close_1d_ago = np.nan
        d_close_5d_ago = np.nan
        if d is not None and not d.empty and "close" in d:
            ds = d["close"].dropna()
            if len(ds) >= 2:
                d_close_1d_ago = float(ds.iloc[-2])
            if len(ds) >= 6:
                d_close_5d_ago = float(ds.iloc[-6])

        chg_1d = pct_change(d_close, d_close_1d_ago)
        chg_5d = pct_change(d_close, d_close_5d_ago)

        # intraday last 15m change
        chg_15m = np.nan
        if i is not None and not i.empty and "close" in i:
            iser = i["close"].dropna()
            if len(iser) >= 2:
                prev_i = float(iser.iloc[-2])
                chg_15m = pct_change(i_close, prev_i)

        # z-score on daily close (20d)
        z20 = np.nan
        if d is not None and not d.empty and "close" in d:
            z20 = zscore_last(d["close"], window=20)

        rows.append(
            {
                "symbol": name,
                "daily_close": d_close,
                "daily_%chg_1d": chg_1d,
                "daily_%chg_5d": chg_5d,
                "intraday_close_15m": i_close,
                "intraday_%chg_last15m": chg_15m,
                "zscore_20d": z20,
            }
        )

    return pd.DataFrame(rows)

# ===== レジーム判定（VIX主導＋円/先物で補強） =====
def eval_regime(feat: pd.DataFrame):
    def row(sym):
        x = feat[feat["symbol"] == sym]
        return x.iloc[0] if len(x) else None

    vix = row("VIX")
    usdjpy = row("USDJPY")
    nfut = row("NIKKEI_FUT")

    base = "NORMAL"
    if vix is not None and not np.isnan(vix["daily_close"]):
        if float(vix["daily_close"]) >= VIX_LEVEL_CRISIS:
            base = "CRISIS"
        elif float(vix["daily_close"]) >= VIX_LEVEL_ALERT:
            base = "ALERT"

    spike_flag = False
    if vix is not None and not np.isnan(vix["intraday_%chg_last15m"]):
        spike_flag = float(vix["intraday_%chg_last15m"]) >= VIX_SPIKE_15M_PCT

    riskoff_flags = []
    if (
        usdjpy is not None
        and not np.isnan(usdjpy["daily_%chg_1d"])
        and float(usdjpy["daily_%chg_1d"]) <= USDJPY_DROP_1D_PCT
    ):
        riskoff_flags.append("USDJPY(円高)")
    if (
        nfut is not None
        and not np.isnan(nfut["daily_%chg_1d"])
        and float(nfut["daily_%chg_1d"]) <= NIKKEI_FUT_DROP_1D_PCT
    ):
        riskoff_flags.append("NIKKEI_FUT下落")

    final_regime = base
    reasons = []
    if vix is not None:
        reasons.append(f"VIX={fp(vix['daily_close'],2)}")
    if spike_flag and vix is not None:
        reasons.append(f"VIX15m+{fp(vix['intraday_%chg_last15m'],2)}%")
    if riskoff_flags:
        reasons.append(" / ".join(riskoff_flags))

    if base == "NORMAL":
        if spike_flag or len(riskoff_flags) >= 2:
            final_regime = "ALERT"
    elif base == "ALERT":
        if spike_flag and len(riskoff_flags) >= 1:
            final_regime = "CRISIS"

    return final_regime, " | ".join(reasons) if reasons else "-"

# ===== テーブル文字列（今のままを維持） =====
def build_table_text(feat: pd.DataFrame) -> str:
    ts = now_jst().strftime("%m/%d %H:%M")
    regime, reason = eval_regime(feat)

    title = f"【Market Regime Monitor】{ts}  Regime={regime}"
    head = "symbol      daily_close  1d%     5d%     intra_close  15m%    z20"

    lines = [title, f"Reason: {reason}", head]

    order = ["VIX", "USDJPY", "NIKKEI", "NIKKEI_FUT", "SPX"]
    for sym in order:
        r = feat[feat["symbol"] == sym]
        if r.empty:
            lines.append(f"{sym:<10}  (no data)")
            continue
        r = r.iloc[0]
        lines.append(
            f"{sym:<10}  "
            f"{fp(r['daily_close'],2):>10}  "
            f"{fp(r['daily_%chg_1d'],3):>7}  "
            f"{fp(r['daily_%chg_5d'],3):>7}  "
            f"{fp(r['intraday_close_15m'],2):>10}  "
            f"{fp(r['intraday_%chg_last15m'],3):>7}  "
            f"{fp(r['zscore_20d'],3):>6}"
        )
    return "\n".join(lines)

# ===== Google News RSS =====
def _rss_locale(lang: str):
    lang = (lang or "ja").lower()
    if lang.startswith("ja"):
        return {"hl": "ja", "gl": "JP", "ceid": "JP:ja"}
    # default to US English
    return {"hl": "en-US", "gl": "US", "ceid": "US:en"}

def build_google_news_rss_url(query: str, lang: str = "ja") -> str:
    loc = _rss_locale(lang)
    q = quote_plus(query.strip())
    return f"https://news.google.com/rss/search?q={q}&hl={loc['hl']}&gl={loc['gl']}&ceid={loc['ceid']}"

def parse_rss(xml_text: str):
    # minimal RSS parser (no external deps)
    items = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items
    channel = root.find("channel")
    if channel is None:
        channel = root.find("{http://www.w3.org/2005/Atom}channel")
    if channel is None:
        return items

    for it in channel.findall("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        source_el = it.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        desc = (it.findtext("description") or "").strip()
        items.append({"title": title, "link": link, "pubDate": pub, "source": source, "description": desc})
    return items

def fetch_google_news_rss(url: str, limit: int = 8):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code >= 300:
            return []
        items = parse_rss(r.text)
        return items[: max(0, limit)]
    except Exception:
        return []

def dedup_items(items):
    seen = set()
    out = []
    for it in items:
        key = (it.get("link") or "") or (it.get("title") or "")
        key = key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

# ===== OpenAI (Responses API) =====
def _openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SEC)

def _safe_json_extract(text: str):
    """
    モデルがJSON以外を混ぜても復旧できるようにする。
    - まずは全体をjson.loads
    - ダメなら最初の {...} or [...] ブロックを拾ってjson.loads
    """
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def ai_propose_rss_queries(feat: pd.DataFrame, regime: str):
    """
    数字を見て「取りに行くべき地政学系クエリ」をAIに決めてもらう
    - 日英ミックス、ただし最終要約は日本語
    """
    client = _openai_client()
    if client is None:
        return []

    # 指標をコンパクトに渡す（コスト節約）
    compact = feat.copy()
    for c in compact.columns:
        if c != "symbol":
            compact[c] = pd.to_numeric(compact[c], errors="coerce")
    payload = compact.to_dict(orient="records")

    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "あなたはマーケット監視AI。与えられた指標（VIX/円/日経先物/SPXなど）から、"
                    "今の相場変動に効きやすい『地政学/マクロ』ニュースを探すためのGoogle News RSS検索クエリを提案して。"
                    "\n\n要件:\n"
                    f"- 出力はJSONのみ\n- クエリ数は最大{MAX_RSS_QUERIES}\n"
                    "- 日英ミックス（日本語クエリと英語クエリを混ぜる）\n"
                    "- “今の数字と関係が薄い”一般論は避ける\n"
                    "- 例: 原油/中東/紅海/台湾/ロシア・ウクライナ/制裁/海運/米金利/Fed/米雇用など\n"
                    "\nJSONスキーマ:\n"
                    "{\n"
                    '  "queries": [\n'
                    '    {"q": "search keywords", "lang": "ja|en", "why": "一言理由"}\n'
                    "  ]\n"
                    "}\n"
                    "\n入力データ:\n"
                    f"Regime={regime}\n"
                    f"features={json.dumps(payload, ensure_ascii=False)}\n"
                ),
            }
        ],
    }

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[prompt],
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        data = _safe_json_extract(resp.output_text)
        if not data or "queries" not in data:
            return []
        queries = data.get("queries") or []
        # sanitize
        out = []
        for it in queries[:MAX_RSS_QUERIES]:
            q = (it.get("q") or "").strip()
            lang = (it.get("lang") or "").strip().lower()
            why = (it.get("why") or "").strip()
            if not q:
                continue
            if lang not in ("ja", "en"):
                lang = RSS_LANG_PRIMARY
            out.append({"q": q, "lang": lang, "why": why})
        return out
    except Exception as e:
        print(f"[WARN] OpenAI propose queries failed: {e}", file=sys.stderr)
        return []

def ai_build_messages(feat: pd.DataFrame, regime: str, reason_line: str, news_items: list):
    """
    3通にするための本文を作る（1通目＋2通目）。3通目はテーブルをそのまま別送。
    """
    client = _openai_client()
    if client is None:
        return None

    compact = feat.copy()
    for c in compact.columns:
        if c != "symbol":
            compact[c] = pd.to_numeric(compact[c], errors="coerce")
    payload = compact.to_dict(orient="records")

    # RSS itemをコンパクト化
    slim_items = []
    for it in news_items[: max(NEWS_PICK_MAX, 1)]:
        slim_items.append(
            {
                "title": it.get("title", ""),
                "source": it.get("source", ""),
                "link": it.get("link", ""),
                "pubDate": it.get("pubDate", ""),
            }
        )

    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "あなたはマーケット監視の解釈担当。出力は日本語。"
                    "ただし、ニュースの見出しは元の言語のままでもよい（=日英ミックス）。"
                    "\n\n目的:\n"
                    "1) 指標の変化を解釈して“いまの結論”と“監視ポイント”を短く書く\n"
                    "2) 与えられたニュース候補から地政学/マクロとして重要なものを最大3〜5本に絞り、"
                    "  見出し（ソース）＋一言要約＋『効く可能性がある理由』＋URL を出す\n"
                    "\n\n制約:\n"
                    "- 1通目は短く（1〜3行＋監視ポイント1〜3点）\n"
                    "- 2通目は3〜5本。各行は短く。\n"
                    "- 推測は控えめに。数字→筋の通る範囲で。\n"
                    "- 出力はJSONのみ。\n"
                    "\nJSONスキーマ:\n"
                    "{\n"
                    '  "msg1": "string",\n'
                    '  "msg2": "string"\n'
                    "}\n"
                    "\n入力:\n"
                    f"Regime={regime}\n"
                    f"ReasonLine={reason_line}\n"
                    f"features={json.dumps(payload, ensure_ascii=False)}\n"
                    f"news_candidates={json.dumps(slim_items, ensure_ascii=False)}\n"
                ),
            }
        ],
    }

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[prompt],
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        data = _safe_json_extract(resp.output_text)
        if not data or "msg1" not in data or "msg2" not in data:
            return None
        return {"msg1": str(data["msg1"]).strip(), "msg2": str(data["msg2"]).strip()}
    except Exception as e:
        print(f"[WARN] OpenAI build messages failed: {e}", file=sys.stderr)
        return None

# ===== 通知（3通：解釈 → ニュース → テーブル） =====
def notify_with_ai(feat: pd.DataFrame):
    regime, reason_line = eval_regime(feat)
    ts = now_jst().strftime("%m/%d %H:%M")

    table_text = build_table_text(feat)

    # 節約：NORMAL時はAIを回さない（AI_ONLY_ON_ALERT=1）
    if AI_ONLY_ON_ALERT and regime == "NORMAL":
        for part in chunk_text(table_text):
            discord_send_text(part)
        return

    # AIを使わない場合は今まで通り（テーブルのみ）
    if not (ENABLE_AI and OPENAI_API_KEY):
        for part in chunk_text(table_text):
            discord_send_text(part)
        return

    # (A) AIに「どのニュースを取りに行くか」を決めさせる
    queries = ai_propose_rss_queries(feat, regime=regime)

    # クエリが取れなかったら、最低限の保険（固定セット）
    if not queries:
        queries = [
            {"q": "中東 原油 供給 リスク", "lang": RSS_LANG_PRIMARY, "why": "risk-off時の定番"},
            {"q": "Taiwan China military tension", "lang": RSS_LANG_SECONDARY, "why": "地政学リスク"},
            {"q": "Fed rates inflation risk", "lang": RSS_LANG_SECONDARY, "why": "米金利/株式連動"},
        ][:MAX_RSS_QUERIES]

    # (B) RSSを取得
    all_items = []
    rss_urls = []
    for q in queries[:MAX_RSS_QUERIES]:
        url = build_google_news_rss_url(q["q"], lang=q.get("lang", RSS_LANG_PRIMARY))
        rss_urls.append({"url": url, "q": q["q"], "why": q.get("why", "")})
        all_items.extend(fetch_google_news_rss(url, limit=RSS_ITEMS_PER_QUERY))

    all_items = dedup_items(all_items)

    # (C) AIで「1通目＋2通目」を生成（3通目はテーブル）
    msgs = ai_build_messages(
        feat=feat,
        regime=regime,
        reason_line=reason_line,
        news_items=all_items[: max(NEWS_PICK_MAX, 1)],
    )

    # msg1 fallback（AI失敗）
    if not msgs:
        msg1 = f"【Regime速報】{ts} Regime={regime}\n{reason_line}\n監視: VIX/ドル円/日経先物の次の1〜2本（15m）"
        # msg2 fallback：RSS URLだけ出す
        lines = ["【地政学ニュース探索】RSS候補（クリック可）"]
        for it in rss_urls:
            why = f" - {it['why']}" if it.get("why") else ""
            lines.append(f"- {it['q']}{why}\n  {it['url']}")
        msg2 = "\n".join(lines)
    else:
        msg1 = f"【Regime速報】{ts}\n{msgs['msg1']}".strip()
        # ついでに「AIが見に行ったRSS」も末尾に付ける（透明性）
        lines = [msgs["msg2"].strip(), "", "（参照RSS）"]
        for it in rss_urls:
            lines.append(f"- {it['q']}  {it['url']}")
        msg2 = "\n".join(lines).strip()

    # 送信（1通目→2通目→3通目）
    for part in chunk_text(msg1):
        discord_send_text(part)
    for part in chunk_text(msg2):
        discord_send_text(part)
    for part in chunk_text(table_text):
        discord_send_text(part)

def main():
    now = now_jst()
    # 指数監視なので週末は基本スキップ（FORCE_RUN=1で強制）
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} weekend (FORCE_RUN=1 to override)")
        return

    feat = build_features()

    # 取得失敗でも落とさず通知
    if feat is None or feat.empty:
        discord_send_text(f"【Market Regime Monitor】{now_jst():%m/%d %H:%M}\nデータ取得失敗")
        return

    # CSVも出しておく（ActionsでArtifact化したい場合に便利）
    out_csv = os.getenv("OUT_CSV", "regime_features.csv")
    feat.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved: {out_csv}")

    notify_with_ai(feat)

if __name__ == "__main__":
    main()
