# scripts/fetch_nse_list.py
import csv, json, os, requests

CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
OUT_DIR = "app/static/app/data"
OUT_FILE = os.path.join(OUT_DIR, "nse_tickers.json")
os.makedirs(OUT_DIR, exist_ok=True)

headers = {"User-Agent":"Mozilla/5.0","Referer":"https://www.nseindia.com"}

r = requests.get(CSV_URL, headers=headers, timeout=30)
r.raise_for_status()
text = r.text.splitlines()
reader = csv.DictReader(text)
out = []
for row in reader:
    symbol = (row.get('SYMBOL') or row.get('Symbol') or '').strip()
    name = (row.get('NAME OF COMPANY') or row.get('NAME') or '').strip()
    series = (row.get('SERIES') or '').strip()
    if not symbol: continue
    label = f"{symbol} — {name}" if name else symbol
    out.append({"label": label, "value": symbol, "meta": series or "EQ"})
out = sorted(out, key=lambda x: x['value'].lower())
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("Wrote", OUT_FILE, "entries:", len(out))
