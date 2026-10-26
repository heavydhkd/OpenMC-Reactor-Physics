import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 确保能导入自定义的物理模型类 (src/models/pin_cell.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pin_cell import VeraPinCell

def main():
    # ==========================================
    # 模块 1: 加载数据
    # ==========================================
    csv_path = os.path.join(os.path.dirname(__file__), '../data/processed/vera_p1_dataset.csv')
    df = pd.read_csv(csv_path)
    
    # 过滤掉失败算例
    df = df[df['status'] == 'success'].copy()
    
    X = df[['enrichment', 'boron_ppm', 'clad_r']].values
    y = df['k_eff'].values
    
    # 划分训练集和测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据标准化 (极其重要)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"数据集划分完毕: 训练集 {len(X_train)} 组, 测试集 {len(X_test)} 组")
    
    # ==========================================
    # 模块 2: 训练模型 (Kriging 与 BP)
    # ==========================================
    print("正在训练 Kriging 模型...")
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr.fit(X_train_scaled, y_train)
    y_pred_gpr, y_std_gpr = gpr.predict(X_test_scaled, return_std=True)
    
    print("正在训练 BP 神经网络...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, early_stopping=True, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    
    # 性能评估
    def calc_pcm_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred)) * 1e5
        
    print("\n--- 模型性能评估 (基于测试集) ---")
    print(f"Kriging RMSE: {calc_pcm_rmse(y_test, y_pred_gpr):.2f} pcm, R2: {r2_score(y_test, y_pred_gpr):.4f}")
    print(f"BP NN   RMSE: {calc_pcm_rmse(y_test, y_pred_mlp):.2f} pcm, R2: {r2_score(y_test, y_pred_mlp):.4f}")
    
    # ==========================================
    # 模块 3: 画图对比
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    min_val, max_val = min(y_test.min(), y_pred_gpr.min()), max(y_test.max(), y_pred_gpr.max())
    ideal_x = np.linspace(min_val, max_val, 100)
    
    ax1.errorbar(y_test, y_pred_gpr, yerr=1.96*y_std_gpr, fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Prediction')
    ax1.plot(ideal_x, ideal_x, 'r--', label='Ideal (y=x)')
    ax1.set_xlabel(r'True \(k_{eff}\)')
    ax1.set_ylabel(r'Predicted \(k_{eff}\)')
    ax1.set_title('Kriging (Gaussian Process)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.scatter(y_test, y_pred_mlp, color='green', label='Prediction')
    ax2.plot(ideal_x, ideal_x, 'r--', label='Ideal (y=x)')
    ax2.set_xlabel(r'True \(k_{eff}\)')
    ax2.set_ylabel(r'Predicted \(k_{eff}\)')
    ax2.set_title('BP Neural Network')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(__file__), '../results/figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'prediction_comparison.png')
    plt.savefig(fig_path, dpi=300)
    print(f"\n对比图已保存至: {os.path.abspath(fig_path)}")


    # ==========================================
    # 模块 4: 智能寻优 (利用代理模型反向设计)
    # ==========================================
    print("\n" + "="*50)
    print("开启智能寻优：寻找固定包壳尺寸下，k_eff=1.0 的参数组合")
    print("="*50)

    target_clad_r = 0.475 

    # 目标函数只为了逼近 k_eff=1.0
    def objective(x):
        # 【修复点】：强制将 scipy 传入的数组元素转为标准 float
        enrichment = float(x[0])
        boron = float(x[1])
        
        features = scaler.transform([[enrichment, boron, target_clad_r]])
        k_pred = gpr.predict(features)[0]
        
        # 绝对误差
        return abs(k_pred - 1.0)

            # 初始猜测：给一个可能临界附近的点 (比如富集度2.5%, 硼500ppm)
    x0 = [2.5, 500.0]
    bounds = [(1.5, 5.0), (0.0, 1200.0)]

    # SLSQP 优化求解
    res = opt.minimize(objective, x0, bounds=bounds, method='SLSQP')
    
    best_enr = res.x[0]
    best_boron = res.x[1]
    
    best_features = scaler.transform([[best_enr, best_boron, target_clad_r]])
    best_pred_k, best_pred_std = gpr.predict(best_features, return_std=True)
    
    print(f"✅ 代理模型寻优结果 (耗时不到 0.01 秒):")
    print(f"   燃料富集度: {best_enr:.4f} %")
    print(f"   硼浓度:     {best_boron:.2f} ppm")
    print(f"   预测 k_eff: {best_pred_k[0]:.5f} ± {best_pred_std[0]:.5f}")
    
    # ==========================================
    # 模块 5: 真实物理验证 (高保真 OpenMC 复核)
    # ==========================================
    print("\n正在调用真实 OpenMC 验证代理模型的寻优结果...")
    verify_model = VeraPinCell(enrichment=best_enr, boron_ppm=best_boron, 
                               clad_outer_r=target_clad_r, fuel_temp=900.0, mod_temp=600.0)
    verify_res = verify_model.run_and_extract(output_dir='../data/raw/optimized_case')
    
    true_k = verify_res['k_eff']
    true_std = verify_res['std']
    
    print(f"\n🎯 真实 OpenMC 验证结果:")
    print(f"   真实 k_eff: {true_k:.5f} ± {true_std:.5f}")
    
    diff_pcm = abs(best_pred_k[0] - true_k) * 1e5
    print(f"   代理模型预测误差: {diff_pcm:.2f} pcm")
    
    # ==========================================
    # 模块 6: 主动学习 (数据回流)
    # ==========================================
    print("\n🔄 启动主动学习：将真实计算结果加入数据集，更新模型...")
    
    # 将新算例加入原来的训练集
    new_row = pd.DataFrame([{
        'case_id': 999, 
        'enrichment': best_enr, 
        'boron_ppm': best_boron, 
        'clad_r': target_clad_r, 
        'k_eff': true_k, 
        'std': true_std, 
        'status': 'success'
    }])
    
    # 更新训练集
    df_updated = pd.concat([df, new_row], ignore_index=True)
    X_updated = df_updated[['enrichment', 'boron_ppm', 'clad_r']].values
    y_updated = df_updated['k_eff'].values
    
    # 重新标准化并训练新的模型
    scaler_updated = StandardScaler()
    X_all_scaled = scaler_updated.fit_transform(X_updated)
    
    gpr_updated = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpr_updated.fit(X_all_scaled, y_updated)
    
    print(f"   数据集已扩充至 {len(df_updated)} 组！")
    print("   代理模型已利用新数据完成更新！(这就是主动学习的雏形)")
    print("\n🎉 完整 AI 代理模型 + 寻优 + 数据回流闭环跑通！")

if __name__ == '__main__':
    main()