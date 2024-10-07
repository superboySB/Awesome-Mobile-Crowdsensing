import requests
import json

OPENROUTER_API_KEY = "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"
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        # "HTTP-Referer": f"{YOUR_SITE_URL}",  # Optional, for including your app on openrouter.ai rankings.
        # "X-Title": f"{YOUR_APP_NAME}",  # Optional. Shows in rankings on openrouter.ai.
    },
    data=json.dumps({
        "model": "meta-llama/llama-3.2-3b-instruct:free",  # Optional
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]

    })
)
print(response.text)
