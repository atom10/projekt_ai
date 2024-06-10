import requests
import json
import os

def send_prompt(prompt):
    url = "http://ollama:11434"
    endpoint = "/api/generate"

    payload = {
        "prompt": prompt,
        "model": os.environ.get('LLM_MODEL', '')
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url + endpoint, json=payload, headers=headers)

    if response.status_code == 200:
        responses = []
        for line in response.iter_lines():
            if line:  # Ignore empty lines
                try:
                    data = json.loads(line)
                    responses.append(data.get("response"))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        return ''.join(responses)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None