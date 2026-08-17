import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
URL = 'https://ucdpapi.pcr.uu.se/api/gedevents/26.1'

# The public API documentation shows this endpoint and version.
# If your account requires a token, set UCDP_TOKEN in the environment.
headers = {}
import os
if os.getenv('UCDP_TOKEN'):
    headers['x-ucdp-access-token'] = os.environ['UCDP_TOKEN']

r = requests.get(URL, params={'pagesize': 1000, 'page': 1}, headers=headers, timeout=60)
r.raise_for_status()
obj = r.json()
(RAW / 'gedevents_26.1_sample.json').write_text(json.dumps(obj), encoding='utf-8')
print('Downloaded', len(obj.get('Result', [])), 'events; total:', obj.get('TotalCount'))
