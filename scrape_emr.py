import json, time, requests, hashlib
from bs4 import BeautifulSoup
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}

URLS = [
    "https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-troubleshoot.html",
    "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-troubleshoot-errors-spark.html",
    "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-troubleshoot-failed.html",
    "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-troubleshoot-errors-resources.html",
    "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-troubleshoot-errors-io.html",
    "https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-troubleshoot-slow.html",
]

def scrape_page(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()

    main = soup.find("div", {"id": "main-content"}) or soup.find("main") or soup.body
    text = "\n".join(l.strip() for l in main.get_text(separator="\n").splitlines() if l.strip())

    return {
        "id": "emr-" + hashlib.md5(url.encode()).hexdigest()[:10],
        "title": soup.title.get_text(strip=True) if soup.title else url,
        "url": url,
        "text": text[:5000],
        "source": "aws-emr-docs",
        "topic": "spark-troubleshooting"
    }

Path("data").mkdir(exist_ok=True)
docs = []
for url in URLS:
    print(f"Fetching {url}")
    try:
        doc = scrape_page(url)
        if len(doc["text"]) > 100:
            docs.append(doc)
        time.sleep(1)
    except Exception as e:
        print(f"  SKIP: {e}")

print(f"Scraped {len(docs)} EMR pages")
with open("data/emr_spark.json", "w") as f:
    json.dump(docs, f, indent=2)