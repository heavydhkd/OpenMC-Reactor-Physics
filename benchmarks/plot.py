import matplotlib.pyplot as plt

# 将你刚才跑出的数据填入
enrichments = [1.6, 2.4, 3.2, 4.95]
keffs = [1.22571, 1.33484, 1.40286, 1.48882]
errors = [0.00294, 0.00319, 0.00259, 0.00297]

plt.figure(figsize=(8, 6))
# 绘制带误差棒的折线图
plt.errorbar(enrichments, keffs, yerr=errors, fmt='o-', capsize=5, color='b', ecolor='r', elinewidth=2, markersize=8)

# 添加轴标签和标题
plt.xlabel('U-235 Enrichment (%)', fontsize=14)
plt.ylabel('k-effective', fontsize=14)
plt.title('PWR Pin Cell: k-effective vs Enrichment', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('k_eff_vs_enrichment.png', dpi=300, bbox_inches='tight')
print("图片保存成功！文件名: k_eff_vs_enrichment.png")
