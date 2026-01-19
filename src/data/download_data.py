import requests
import os

# create e2e_data
current_dir = os.path.dirname(os.path.abspath(__file__))
e2e_dir = os.path.join(current_dir, "e2e_data")
os.makedirs(e2e_dir, exist_ok=True)

files = [
    ("train.csv", "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/train-fixed.no-ol.csv"),
    ("dev.csv", "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/devel-fixed.no-ol.csv"),
    ("test.csv", "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/test-fixed.csv")
]

for filename, url in files:
    save_path = os.path.join(e2e_dir, filename)
    print(f"downloading {filename}...")
    try:
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"save to: {save_path}")
    except Exception as e:
        print(f"error: {e}")

print(f"\n Succeed")