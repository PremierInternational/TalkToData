import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'mistral',
        'prompt': 'tell me 10 employerID business terms only',
        'stream': False  # 👈 disable streaming to make JSON parsing easy
    }
)

print(response.json()['response'])  # ✅ Now this should work
