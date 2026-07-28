"""
Scrapes Databricks KB articles related to Spark/Jobs and saves to data/documents.json
Run: uv run python scrape_kb.py
"""

import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://kb.databricks.com"
ALL_ARTICLES_URL = f"{BASE_URL}/en_US/all-articles"

SPARK_KEYWORDS = [
    "spark", "executor", "driver", "shuffle", "oom", "out of memory",
    "job fail", "job abort", "stage fail", "task fail", "cluster",
    "fetchfailed", "heap", "serializ", "partition", "rdd", "dataframe",
    "streaming", "pyspark", "scala spark"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"
}


def get_article_links():
    """Fetch all article links from the KB index page."""
    print("Fetching article list...")
    resp = requests.get(ALL_ARTICLES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # KB articles are at /en_US/... or relative paths
        if "/en_US/" in href or (href.startswith("/") and len(href) > 10):
            full_url = href if href.startswith("http") else BASE_URL + href
            title = a.get_text(strip=True)
            if title:
                links.append({"url": full_url, "title": title})

    # Deduplicate by URL
    seen = set()
    unique = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique.append(link)

    print(f"Found {len(unique)} unique article links")
    return unique


def is_spark_related(title, text=""):
    combined = (title + " " + text).lower()
    return any(kw in combined for kw in SPARK_KEYWORDS)


def scrape_article(url, title):
    """Fetch a single KB article and extract its content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav/footer/sidebar noise
        for tag in soup.find_all(["nav", "footer", "header", "script", "style"]):
            tag.decompose()

        # Main content area
        content_div = (
            soup.find("div", class_="article-body") or
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"id": "content"})
        )

        text = content_div.get_text(separator="\n", strip=True) if content_div else soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        # Generate stable ID from URL
        doc_id = "spark-kb-" + hashlib.md5(url.encode()).hexdigest()[:10]

        return {
            "id": doc_id,
            "title": title,
            "url": url,
            "text": text[:5000],  # cap at 5000 chars per doc
            "source": "databricks-kb",
            "topic": "spark-troubleshooting"
        }

    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None


def main():
    Path("data").mkdir(exist_ok=True)

    all_links = get_article_links()

    # First pass: filter by title
    candidates = [l for l in all_links if is_spark_related(l["title"])]
    print(f"{len(candidates)} Spark-related articles by title")

    documents = []
    for i, link in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] {link['title'][:70]}")
        doc = scrape_article(link["url"], link["title"])
        if doc and len(doc["text"]) > 100:
            documents.append(doc)
        time.sleep(0.5)  # be polite

    print(f"\nScraped {len(documents)} documents")

    with open("data/documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print("Saved to data/documents.json")

    # Preview
    print("\nSample doc IDs and titles:")
    for doc in documents[:5]:
        print(f"  {doc['id']} — {doc['title']}")


if __name__ == "__main__":
    main()