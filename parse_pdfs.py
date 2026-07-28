"""
Parse DE scenario PDFs into documents.json
Usage: python parse_pdfs.py <uploads_dir> <output_file>
Example: python parse_pdfs.py C:/Users/.../uploads data/pdf_scenarios.json
"""

import json
import re
import subprocess
import hashlib
import sys
from pathlib import Path

PDF_FILES = {
    "1_data_pipeline_failures.pdf": "data-pipeline-failures",
    "2_performance_latency_issues.pdf": "performance-latency",
    "3_streaming_kafka_incident.pdf": "streaming-kafka",
    "4_data_quality_trust_breaks.pdf": "data-quality",
    "5_cloud_cost_resource_explosions.pdf": "cloud-cost-resources",
    "6_orchestration_scheduling_issues.pdf": "orchestration-scheduling",
    "7_schema_evolution_problems.pdf": "schema-evolution",
    "8_backfills_reprocessing_recovery.pdf": "backfills-reprocessing",
    "9_security_access_compliance_incidents.pdf": "security-access",
    "10_stakeholder_process_people_pressure.pdf": "stakeholder-process",
}

SECTION_HEADERS = [
    "Problem Statement",
    "Clarifying Questions",
    "Clarifying Information",
    "Confirmed Facts",
    "Key Observation",
    "Root Cause Analysis",
    "Investigation & Root Cause",
    "Solution:",
    "Final Resolution",
    "Key Learning",
    "Core Principle Reinforced",
    "Why This",
    "Expected vs Actual",
    "What the",
    "What Teams",
    "What Kafka",
    "What the System",
    "What the Organization",
]


def pdf_to_text(pdf_path):
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True, text=True
    )
    return result.stdout


def split_into_scenarios(text):
    """Split full PDF text into individual scenarios, skipping the TOC."""
    # Find where actual content starts (first "Scenario N\n" after TOC)
    # TOC entries look like "Scenario 1..........5" (with dots)
    # Real scenarios look like "Scenario 1\n\nTitle"
    lines = text.split("\n")

    # Find first real scenario heading (not TOC — no dots/numbers at end)
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^Scenario\s+\d+\s*$', stripped):
            content_start = i
            break

    content_lines = lines[content_start:]
    content = "\n".join(content_lines)

    # Split on "Scenario N" pattern (standalone line)
    parts = re.split(r'\n(?=Scenario\s+\d+\s*\n)', content)

    scenarios = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 100:
            continue
        scenarios.append(part)

    return scenarios


def extract_title(scenario_text):
    """Extract scenario title — line after 'Scenario N'."""
    lines = [l.strip() for l in scenario_text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if re.match(r'^Scenario\s+\d+$', line):
            if i + 1 < len(lines):
                return lines[i + 1]
    return lines[0] if lines else "Unknown"


def extract_section(text, section_name):
    """Extract text under a section header until the next header."""
    alternation = '|'.join(re.escape(h) for h in SECTION_HEADERS)
    pattern = re.compile(
        rf'{re.escape(section_name)}[\s\S]*?(?=(?:{alternation})|$)',
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        content = match.group(0)
        # Remove the header itself
        content = re.sub(rf'^{re.escape(section_name)}\s*', '', content, flags=re.IGNORECASE)
        return content.strip()[:2000]
    return ""


def parse_scenario(scenario_text, category, scenario_num):
    title = extract_title(scenario_text)

    problem = extract_section(scenario_text, "Problem Statement")
    root_cause = (
        extract_section(scenario_text, "Root Cause Analysis") or
        extract_section(scenario_text, "Investigation & Root Cause")
    )
    resolution = extract_section(scenario_text, "Final Resolution")
    learnings = (
        extract_section(scenario_text, "Key Learning") or
        extract_section(scenario_text, "Key Learnings")
    )
    principle = extract_section(scenario_text, "Core Principle Reinforced")

    # Full text for search (capped)
    full_text = "\n\n".join(filter(None, [
        f"Problem: {problem}",
        f"Root Cause: {root_cause}",
        f"Resolution: {resolution}",
        f"Key Learnings: {learnings}",
        f"Core Principle: {principle}",
    ]))[:5000]

    doc_id = f"pdf-{category}-{scenario_num}-" + hashlib.md5(title.encode()).hexdigest()[:6]

    return {
        "id": doc_id,
        "title": title,
        "category": category,
        "source": "de-production-scenarios",
        "problem": problem,
        "root_cause": root_cause,
        "resolution": resolution,
        "key_learnings": learnings,
        "core_principle": principle,
        "text": full_text,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_pdfs.py <uploads_dir> <output_json>")
        sys.exit(1)

    uploads_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_docs = []

    for filename, category in PDF_FILES.items():
        pdf_path = uploads_dir / filename
        if not pdf_path.exists():
            print(f"SKIP (not found): {filename}")
            continue

        print(f"Parsing {filename}...")
        text = pdf_to_text(pdf_path)
        scenarios = split_into_scenarios(text)
        print(f"  Found {len(scenarios)} scenarios")

        for i, scenario_text in enumerate(scenarios, 1):
            doc = parse_scenario(scenario_text, category, i)
            if doc["problem"] or doc["resolution"]:
                all_docs.append(doc)
                print(f"  [{i}] {doc['title'][:60]}")

    print(f"\nTotal documents: {len(all_docs)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
