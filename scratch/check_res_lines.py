lines = open("generate_all_figures_standalone.py", encoding="utf-8").readlines()
print("Lines 515-585:")
for i in range(515, 585):
    if i < len(lines):
        print(f"{i}: {lines[i]}", end="")
