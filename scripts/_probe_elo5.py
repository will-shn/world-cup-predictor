import re
import requests

t = requests.get("https://www.eloratings.net/scripts/ratings.js", timeout=30).text
m = re.search(r"function pageName\([^)]*\)\s*\{[^}]+\}", t)
print(m.group(0) if m else "not found")
# try broader
idx = t.find("function pageName")
print(t[idx : idx + 200])
