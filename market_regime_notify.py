import os
import requests
import feedparser
from datetime import datetime, timezone
from openai import OpenAI

# =========================
# ENV
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_MAX_AGE_HOURS = int(os.getenv("NEWS_MAX_AGE_HOURS", "6"))
MAX_NEWS_ITEMS = 30
MAX_PICKED = 5

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# RSS SOURCES（量重視）
# =========================
RSS_URLS = [
    "https://news.google.com/rss/search?q=USDJPY+intervention+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=yen+intervention+risk+Reuters+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=US+interest+rates+market+impact+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Middle+East+oil+geopolitical+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Taiwan+China+military+market+when:6h&hl=en-US&gl=US&ceid=US:en",
]

# =========================
# FETCH RSS
# =========================
def collect_articles():
    articles = []
    now = datetime.now(timezone.utc)

    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            if not hasattr(e, "published_parsed"):
                continue

            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            hours_ago = (now - published).total_seconds() / 3600
            if hours_ago > NEWS_MAX_AGE_HOURS:
                continue

            articles.append({
                "id": f"{e.get('source', {}).get('title', 'news')}_{len(articles)}",
                "title": e.title,
                "source": e.get("source", {}).get("title", "Unknown"),
                "published_hours_ago": round(hours_ago, 1),
                "url": e.link,
                "snippet": e.summary[:300]
            })

            if len(articles) >= MAX_NEWS_ITEMS:
                return articles

    return articles

# =========================
# AI ANALYSIS
# =========================
def analyze_with_ai(articles):
    prompt = f"""
あなたはプロのマーケットアナリストです。

以下は直近ニュースの記事一覧です。
それぞれについて：
1. 株価・為替・金利・VIXに効く観点で日本語1〜2行で要約
2. 重要度を1〜5で評価

最後に、最も市場影響が大きいものを最大{MAX_PICKED}件選んでください。

出力は必ずJSONのみ。

記事一覧：
{articles}
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
        temperature=0.2
    )

    return res.choices[0].message.content

# =========================
# MAIN
# =========================
def run_ai_news_block():
    articles = collect_articles()
    if not articles:
        return None, "直近ニュースを取得できませんでした。"

    try:
        ai_result = analyze_with_ai(articles)
        return ai_result, None
    except Exception as e:
        return None, f"AI解析失敗: {e}"

# =========================
# 実行（既存通知の前に差し込む）
# =========================
if __name__ == "__main__":
    result, error = run_ai_news_block()
    if error:
        print("[AI NEWS ERROR]", error)
    else:
        print("[AI NEWS RESULT]")
        print(result)
