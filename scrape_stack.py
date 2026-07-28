# save as scrape_stackoverflow.py
import json, time, requests
from pathlib import Path

def fetch_spark_questions(pages=10):
    docs = []
    for page in range(1, pages + 1):
        resp = requests.get(
            "https://api.stackexchange.com/2.3/questions",
            params={
                "order": "desc", "sort": "votes", "tagged": "apache-spark",
                "site": "stackoverflow", "filter": "withbody",
                "pagesize": 100, "page": page,
                "min": 5,  # min score — filters junk
            }
        )
        data = resp.json()
        for q in data.get("items", []):
            if not q.get("is_answered") or not q.get("accepted_answer_id"):
                continue
            docs.append({
                "id": f"so-{q['question_id']}",
                "title": q["title"],
                "body": q.get("body", "")[:3000],
                "url": q["link"],
                "source": "stackoverflow",
                "topic": "spark-error",
            })
        print(f"Page {page}: {len(data.get('items', []))} questions")
        if not data.get("has_more"):
            break
        time.sleep(1)
    return docs

Path("data").mkdir(exist_ok=True)
docs = fetch_spark_questions(pages=5)
print(f"Got {len(docs)} SO questions")
with open("data/so_spark.json", "w") as f:
    json.dump(docs, f, indent=2)