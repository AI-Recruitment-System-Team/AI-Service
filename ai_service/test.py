import time
import ollama

start = time.time()

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": 'Return ONLY this JSON: {"ok": true}'
        }
    ],
    format="json",
    options={"temperature": 0}
)

print(response["message"]["content"])
print("Time:", time.time() - start)
response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "/no_think\nReturn only {\"ok\":true}"
        }
    ],
    format="json"
)