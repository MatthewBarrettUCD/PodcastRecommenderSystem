import requests
from pprint import pprint

#r = requests.get('https://listen-api.listennotes.com/api/v2/search?q=star%20wars&sort_by_date=0&type=episode&offset=0&len_min=10&len_max=30&genre_ids=68%2C82&published_before=1390190241000&published_after=0&only_in=title%2Cdescription&language=English&safe_mode=1')
#r.json()

response = requests.get("https://listen-api.listennotes.com/api/v2/search?sort_by_date=0&type=podcast&offset=0&len_min=0&published_after=0&only_in=title%2Cdescription&language=English&safe_mode=0",
  headers={
    "X-ListenAPI-Key": "d1932d3198964e2891821c9013c9fba0",
  },
)

response.json()
pprint(response.json())