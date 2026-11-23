lines = open("generate_all_figures_standalone.py", encoding="utf-8").readlines()
print(f"Total lines: {len(lines)}")
for i, l in enumerate(lines):
    if l.startswith("def ") or l.startswith("class "):
        print(f"Line {i}: {l.strip()}")
