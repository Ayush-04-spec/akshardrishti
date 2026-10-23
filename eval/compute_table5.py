"""
Compute Table 5: Domain Distribution
Analyzes benchmark/metadata.csv to count pages per domain.
"""
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
metadata_path = ROOT / "benchmark" / "metadata.csv"

print("="*70)
print("TABLE 5: Domain Distribution in Benchmark")
print("="*70)

# Read metadata
with open(metadata_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_pages = len(rows)
print(f"\nTotal benchmark pages: {total_pages}\n")

# Count by domain
domain_counts = Counter(row['domain'] for row in rows)

# Map raw domains to paper categories
domain_mapping = {
    'order_circular': 'Government Orders & Circulars',
    'tender': 'Tenders & Notices',
    'notice': 'Tenders & Notices',
    'report': 'Reports',
    'wide_table': 'Tables & Lists',
    'transfer_list': 'Tables & Lists',
    'form_table': 'Forms',
    'other': 'Other'
}

# Group by paper category
category_counts = Counter()
for domain, count in domain_counts.items():
    category = domain_mapping.get(domain, 'Other')
    category_counts[category] += count

# Sort by count descending
sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

print(f"{'Domain Category':<35} {'Pages':>10} {'Percentage':>12}")
print("-"*70)

for category, count in sorted_categories:
    percentage = (count / total_pages) * 100
    print(f"{category:<35} {count:>10} {percentage:>11.1f}%")

print("-"*70)
print(f"{'TOTAL':<35} {total_pages:>10} {100.0:>11.1f}%")

# Raw domain breakdown
print("\n" + "="*70)
print("Detailed Breakdown by Raw Domain:")
print("="*70)
print(f"{'Domain':<25} {'Count':>10} {'Percentage':>12}")
print("-"*70)

for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_pages) * 100
    print(f"{domain:<25} {count:>10} {percentage:>11.1f}%")

# Language breakdown
lang_counts = Counter(row['language'] for row in rows)
print("\n" + "="*70)
print("Language Distribution:")
print("="*70)
for lang, count in sorted(lang_counts.items()):
    percentage = (count / total_pages) * 100
    print(f"{lang.capitalize():<20} {count:>5} pages ({percentage:>5.1f}%)")

# Save results
import json
results = {
    "total_pages": total_pages,
    "by_category": {cat: {"count": cnt, "percentage": round((cnt/total_pages)*100, 2)} 
                    for cat, cnt in sorted_categories},
    "by_raw_domain": {dom: {"count": cnt, "percentage": round((cnt/total_pages)*100, 2)}
                      for dom, cnt in domain_counts.items()},
    "by_language": {lang: {"count": cnt, "percentage": round((cnt/total_pages)*100, 2)}
                    for lang, cnt in lang_counts.items()}
}

out_path = ROOT / "results" / "table5_domain_distribution.json"
out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f"\nSaved: {out_path}")
