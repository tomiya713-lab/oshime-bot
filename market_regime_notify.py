# market_regime_notify_cause_search.py
# Re-issued copy (cause-search enabled)

import os
import json
import requests
import feedparser
from datetime import datetime, timezone
from openai import OpenAI

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

NEWS_MAX_AGE_HOURS = int(os.environ.get("NEWS_MAX_AGE_HOURS", "6"))
MAX_NEWS_ITEMS = 30
MAX_PICKED_NEWS = 5

client = OpenAI(api_key=OPENAI_API_KEY)

def utc_now():
    return datetime.now(timezone.utc)

def hours_ago(dt):
    return (utc_now() - dt).total_seconds() / 3600

RSS_QUERIES = [
    "USDJPY intervention",
    "FRB chair nomination",
    "US Treasury yield surge",
    "Wall Street plunge",
    "Nikkei futures drop",
    "market volatility spike",
]

RSS_URLS = [
    f"https://news.google.com/rss/search?q={q}&hl=en&gl=US&ceid=US:en"
    for q in RSS_QUERIES
]

def collect_news():
    items = []
    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            if not hasattr(e, "published_parsed"):
                continue
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if hours_ago(published) > NEWS_MAX_AGE_HOURS:
                continue

            items.append({
                "title": e.title,
                "source": e.get("source", {}).get("title", ""),
                "url": e.link,
                "summary": e.get("summary", "")[:400],
                "hours_ago": round(hours_ago(published), 1),
            })

            if len(items) >= MAX_NEWS_ITEMS:
                return items
    return items

def analyze_market_causes(market_snapshot, news_items):
    prompt = f"""
You are a professional market analyst.

Market moves (last 6 hours):
{json.dumps(market_snapshot, ensure_ascii=False, indent=2)}

Candidate news list:
{json.dumps(news_items, ensure_ascii=False, indent=2)}

Tasks:
1. Identify news most likely responsible for the market moves.
2. Select up to {MAX_PICKED_NEWS} articles.
3. For each article:
   - 1–2 line Japanese summary
   - Why it impacts the market (1 line)
   - Affected sectors (max 3, comma-separated, with + / -)
4. Summarize overall market cause (1 paragraph).
5. Give short-term outlook (up/down/volatile).
6. List up to 3 things to watch in the next 1–2 hours.

Output in clean Japanese text.
"""
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    return res.choices[0].message.content

def post_discord(text):
    requests.post(DISCORD_WEBHOOK_URL, json={"content": text}, timeout=10)

def main():
    market_snapshot = {
        "USDJPY": "+1.2円",
        "VIX": "+8%",
        "SPX": "-2.1%",
        "NIKKEI_FUT": "-1.3%",
    }

    news = collect_news()
    if not news:
        post_discord("【AI分析】直近ニュースを取得できませんでした。")
        return

    analysis = analyze_market_causes(market_snapshot, news)
    post_discord(analysis)

if __name__ == "__main__":
    main()
