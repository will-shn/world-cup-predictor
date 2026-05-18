import re
import requests

r = requests.get("https://www.eloratings.net/scripts/ratings.js", timeout=30)
text = r.text
print("len", len(text))
# team display names often appear near .tsv loads
for needle in ["World.tsv", ".tsv", "countries", "Teams"]:
    print(needle, text.count(needle))

# CLDR locale data sometimes has code -> name
for m in re.finditer(r"([A-Z]{2,3})['\"]:\s*['\"]([^'\"]{3,40})['\"]", text):
    if m.start() < 200000:
        pass
hits = re.findall(r"\{[^}]{0,80}code[^}]{0,200}\}", text[:100000])
print("code objects sample", hits[:3])

# search for England, Brazil
for name in ["England", "Brazil", "Spain", "United States"]:
    idx = text.find(name)
    print(name, "at", idx, repr(text[idx:idx+80]) if idx >= 0 else "")
