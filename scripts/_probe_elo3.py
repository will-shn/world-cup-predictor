import requests

t = requests.get("https://www.eloratings.net/scripts/ratings.js", timeout=30).text
# extract function that loads World data
idx = t.find("teamDictionary")
print("teamDictionary context:")
print(t[idx : idx + 2500])
