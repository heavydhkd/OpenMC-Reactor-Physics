import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# 读取 scan.py 生成的扫描结果（与脚本同目录的 scan_results.csv）
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan_results.csv')
if not os.path.exists(csv_path):
    print(f"未找到扫描结果文件: {csv_path}")
    print("请先运行 `python benchmarks/scan.py` 完成扫描。")
    sys.exit(1)

df = pd.read_csv(csv_path)
enrichments = df['enrichment']
keffs = df['k_eff']
errors = df['std']

plt.figure(figsize=(8, 6))
# 绘制带误差棒的折线图
plt.errorbar(enrichments, keffs, yerr=errors, fmt='o-', capsize=5, color='b', ecolor='r', elinewidth=2, markersize=8)

# 添加轴标签和标题
plt.xlabel('U-235 enrichment (wt%)', fontsize=14)
plt.ylabel('$k_{eff}$', fontsize=14)
plt.title('PWR Pin Cell: k-effective vs Enrichment', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('k_eff_vs_enrichment.png', dpi=300, bbox_inches='tight')
print("图片保存成功！文件名: k_eff_vs_enrichment.png")
