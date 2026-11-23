import re, os

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

md_imgs = re.findall(r'!\[.*?\]\((.*?)\)', text)
html_imgs = re.findall(r'src=["\'](.*?)["\']', text)

all_imgs = md_imgs + html_imgs
print(f"Total images referenced: {len(all_imgs)}")

missing = []
for img in all_imgs:
    if img.startswith('http'):
        continue
    path = img.split()[0].replace('/', os.sep)
    if not os.path.exists(path):
        missing.append(img)

if missing:
    print("WARNING: Missing files:")
    for m in missing:
        print(" ", m)
else:
    print(f"SUCCESS: All {len(all_imgs)} image paths exist on disk!")
