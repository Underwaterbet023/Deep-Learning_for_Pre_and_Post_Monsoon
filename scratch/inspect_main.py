lines = open("generate_all_figures_standalone.py", encoding="utf-8").readlines()
for i in range(835, len(lines)):
    print(f"{i}: {lines[i]}", end="")
