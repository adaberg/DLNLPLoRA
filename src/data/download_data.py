# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 20:22:15 2026

@author: Alo
"""

# 放在 src/data/download_data.py
import requests
import os

# 创建e2e_data目录（在data文件夹下）
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
    print(f"下载 {filename}...")
    try:
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 保存到: {save_path}")
    except Exception as e:
        print(f"❌ 失败: {e}")

print(f"\n🎉 完成！路径: {e2e_dir}")