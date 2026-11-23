lines = open("generate_all_figures_standalone.py", encoding="utf-8").readlines()
print("Lines 240-270:")
for i in range(240, 270):
    if i < len(lines):
        print(f"{i}: {lines[i]}", end="")
