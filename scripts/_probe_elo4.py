import requests

t = requests.get("https://www.eloratings.net/scripts/ratings.js", timeout=30).text
for fn in ["buildWorld", "buildTeam", "teamDictionary", "pageName", "menu.tsv"]:
    idx = 0
    while True:
        idx = t.find(fn, idx)
        if idx < 0:
            break
        print("---", fn, "at", idx, "---")
        print(t[idx : idx + 600])
        print()
        idx += len(fn)
