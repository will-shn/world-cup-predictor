import re
import requests

t = requests.get("https://www.eloratings.net/scripts/ratings.js", timeout=30).text
for url in ["Teams.tsv", "Team.tsv", "teams.tsv", "World.tsv", "Tournaments.tsv"]:
    r = requests.head(f"https://www.eloratings.net/{url}", timeout=10)
    print(url, r.status_code)

seen = set(re.findall(r"['\"]([A-Za-z0-9 _-]+\.tsv)['\"]", t))
print("tsv refs in js:", sorted(seen))

# Teams.tsv likely has code -> display name
r = requests.get("https://www.eloratings.net/Teams.tsv", timeout=30)
print("Teams.tsv status", r.status_code, "len", len(r.content))
if r.ok:
    lines = r.text.splitlines()[:8]
    for line in lines:
        print(repr(line[:120]))
