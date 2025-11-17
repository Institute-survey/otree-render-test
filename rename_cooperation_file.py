import os
import re

# === 対象ディレクトリ ===
target_dir = "."  # 同じフォルダで実行する場合は "." のままでOK

# === 正規表現パターン ===
# 例: network0p05 → network0.05
pattern_network = re.compile(r"(network)(\d+(?:\.\d+)?)[p](\d+(?:\.\d+)?)")

# 例: benefit5p0 → benefit5.0
pattern_benefit = re.compile(r"(benefit)(\d+(?:\.\d+)?)[p](\d+(?:\.\d+)?)")

for filename in os.listdir(target_dir):
    if not filename.startswith("cooperation_rates_") or not filename.endswith(".csv"):
        continue

    old_path = os.path.join(target_dir, filename)
    new_name = filename

    # networkX p Y → networkX.Y
    new_name = pattern_network.sub(r"\1\2.\3", new_name)

    # benefitX p Y → benefitX.Y
    new_name = pattern_benefit.sub(r"\1\2.\3", new_name)

    # ファイル名が変わっている場合のみ rename
    if new_name != filename:
        new_path = os.path.join(target_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename}  →  {new_name}")
