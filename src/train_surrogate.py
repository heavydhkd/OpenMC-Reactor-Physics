import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. 加载数据
    csv_path = os.path.join(os.path.dirname(__file__), '../data/processed/vera_p1_dataset.csv')
    df = pd.read_csv(csv_path)
    
    # 过滤掉可能存在的失败算例
    df = df[df['status'] == 'success'].copy()
    
    # 提取特征 (X) 和 目标变量 (y)
    X = df[['enrichment', 'boron_ppm', 'clad_r']].values
    y = df['k_eff'].values
    
    # 2. 划分训练集和测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 数据标准化 (极其重要！因为硼浓度的量级是 1000，而富集度是 5，不标准化神经网络会脑死)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"数据集划分完毕: 训练集 {len(X_train)} 组, 测试集 {len(X_test)} 组")
    
    # ==========================================
    # 模型 A: Kriging (高斯过程回归)
    # ==========================================
    print("正在训练 Kriging 模型...")
    # RBF 处理非线性，WhiteKernel 用来吸收 OpenMC 的统计噪声
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train_scaled, y_train)
    
    # 预测并返回标准差 (置信区间)
    y_pred_gpr, y_std_gpr = gpr.predict(X_test_scaled, return_std=True)
    
    # ==========================================
    # 模型 B: BP 神经网络 (多层感知机)
    # ==========================================
    print("正在训练 BP 神经网络...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, early_stopping=True, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    
    # ==========================================
    # 性能评估
    # ==========================================
    def calc_pcm_rmse(y_true, y_pred):
        # 将绝对误差转换为 pcm (1 pcm = 0.00001)
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse) * 1e5
        
    print("\n--- 模型性能评估 (基于测试集) ---")
    print(f"Kriging RMSE: {calc_pcm_rmse(y_test, y_pred_gpr):.2f} pcm, R2: {r2_score(y_test, y_pred_gpr):.4f}")
    print(f"BP NN   RMSE: {calc_pcm_rmse(y_test, y_pred_mlp):.2f} pcm, R2: {r2_score(y_test, y_pred_mlp):.4f}")
    
    # ==========================================
    # 画图：预测值 vs 真实值
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    min_val, max_val = min(y_test.min(), y_pred_gpr.min()), max(y_test.max(), y_pred_gpr.max())
    ideal_x = np.linspace(min_val, max_val, 100)
    
    # 图 1: Kriging
    ax1.errorbar(y_test, y_pred_gpr, yerr=1.96*y_std_gpr, fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Prediction')
    ax1.plot(ideal_x, ideal_x, 'r--', label='Ideal (y=x)')
    ax1.set_xlabel(r'True \(k_{eff}\)')
    ax1.set_ylabel(r'Predicted \(k_{eff}\)')
    ax1.set_title('Kriging (Gaussian Process)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 图 2: BP NN
    ax2.scatter(y_test, y_pred_mlp, color='green', label='Prediction')
    ax2.plot(ideal_x, ideal_x, 'r--', label='Ideal (y=x)')
    ax2.set_xlabel(r'True \(k_{eff}\)')
    ax2.set_ylabel(r'Predicted \(k_{eff}\)')
    ax2.set_title('BP Neural Network')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 保存图片
    fig_dir = os.path.join(os.path.dirname(__file__), '../results/figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'prediction_comparison.png')
    plt.savefig(fig_path, dpi=300)
    print(f"\n对比图已保存至: {os.path.abspath(fig_path)}")

if __name__ == '__main__':
    main()