import requests
import os

# Read .env to get NCBI_API_KEY if present
key = None
if os.path.exists('.env'):
    with open('.env','r',encoding='utf-8') as fh:
        for line in fh:
            if line.strip().startswith('NCBI_API_KEY='):
                key = line.strip().split('=',1)[1]
                break

params = {'db':'gene','id':'672','retmode':'xml'}
if key:
    params['api_key'] = key
url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
print('Requesting', url)
r = requests.get(url, params=params, timeout=30)
print('Status:', r.status_code)
print(r.text[:4000])
