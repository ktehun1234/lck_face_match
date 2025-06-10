import requests
import urllib.parse
API_KEY = "RGAPI-11b5aa5f-da8b-4889-b21e-e847111ccf47"
REGION = "kr"
summoner_name = "Hide on bush"
encoded = urllib.parse.quote(summoner_name)
url = f"https://{REGION}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{encoded}"
headers = {"X-Riot-Token": API_KEY}

resp = requests.get(url, headers=headers)
print(resp.status_code)
print(resp.text)