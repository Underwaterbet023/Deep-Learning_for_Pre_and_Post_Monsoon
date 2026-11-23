import json

trans = r"C:\Users\Sanskar\.gemini\antigravity-ide\brain\ef343278-5594-4566-9e37-08096ee664ed\.system_generated\logs\transcript.jsonl"
for i, line in enumerate(open(trans, encoding="utf-8")):
    obj = json.loads(line)
    if obj.get("type") == "USER_INPUT":
        print(f"--- User Input at line {i} ---")
        print(obj.get("content"))
