import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# 修复 1：使用新版 SALib 推荐的采样器
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

def main():
    # 1. 载入 5 维数据并训练代理模型
    csv_path = os.path.join(os.path.dirname(__file__), '../data/processed/vera_p1_dataset.csv')
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'success']
    
    feature_cols = ['enrichment', 'boron_ppm', 'clad_r', 'fuel_temp', 'mod_temp']
    X = df[feature_cols].values
    y = df['k_eff'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("正在训练 5 维 Kriging 模型...")
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0]*5) + WhiteKernel(noise_level=1e-4)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    
    # 忽略由于样本较少可能带来的优化器警告，保证终端输出干净
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    gpr.fit(X_scaled, y)
    
    # 2. 设置 Sobol 敏感性分析的参数空间 (必须与 yaml 里的边界一致)
    problem = {
        'num_vars': 5,
        'names': ['Enrichment', 'Boron', 'Clad_R', 'Fuel_Temp', 'Mod_Temp'],
        'bounds': [
            [1.5, 5.0], 
            [0.0, 1200.0], 
            [0.45, 0.49], 
            [600.0, 1200.0], 
            [500.0, 650.0]
        ]
    }
    
    print("正在生成 Sobol 采样矩阵...")
    # 修复 2：使用新版的 sample 方法，并显式声明 calc_second_order=False
    param_values = sobol_sample.sample(problem, 1024, calc_second_order=False)
    
    # 3. 用代理模型瞬间预测所有样本的 k_eff
    print(f"代理模型正在瞬间预测 {len(param_values)} 组样本...")
    param_values_scaled = scaler.transform(param_values)
    Y = gpr.predict(param_values_scaled)
    
    # 4. 计算 Sobol 指数
    print("正在计算 Sobol 敏感性指数...")
    # 修复 3：在 analyze 中同步传入 calc_second_order=False，解决样本量不匹配报错
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    
    # 5. 可视化：画出敏感性柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(len(problem['names']))
    width = 0.35
    
    # S1: 一阶敏感度 (该参数独立的影响)
    # ST: 总敏感度 (包含该参数与其他参数的交互影响)
    ax.bar(indices - width/2, Si['S1'], width, label='First-order (S1)', color='skyblue')
    ax.bar(indices + width/2, Si['ST'], width, label='Total-order (ST)', color='salmon')
    
    ax.set_ylabel('Sobol Index')
    # 修复 4：加入 r 前缀，消除转义字符警告
    ax.set_title(r'Global Sensitivity Analysis of \(k_{eff}\)')
    ax.set_xticks(indices)
    ax.set_xticklabels(problem['names'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '../results/figures/sobol_sensitivity.png')
    plt.savefig(fig_path, dpi=300)
    
    print("\n--- Sobol 敏感性分析结果 ---")
    for i, name in enumerate(problem['names']):
        print(f"{name:10s}: S1 = {Si['S1'][i]:.4f}, ST = {Si['ST'][i]:.4f}")
        
    print(f"\n图表已保存至: {os.path.abspath(fig_path)}")

if __name__ == '__main__':
    main()