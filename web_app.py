import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time
# ==========================================
# 1. 模拟 AI 代理模型（后台运行，不可见）
# ==========================================
# 为了让网页立刻能跑，我们先用代码生成 200 组假数据来训练一个 AI 模型
# （以后你可以替换成用 OpenMC 生成的真数据和你的 Kriging 模型）
@st.cache_data
def train_model():
    np.random.seed(42)
    n_samples = 200
    # 输入特征：富集度(1.5-5.0), 慢化剂密度(0.6-0.8)
    enrichment = np.random.uniform(1.5, 5.0, n_samples)
    density = np.random.uniform(0.6, 0.8, n_samples)
    
    # 模拟一个物理规律：富集度越高 k_eff 越大，密度越高 k_eff 越大（带一点噪声）
    keff = 0.8 + 0.08 * enrichment + 0.3 * density + np.random.normal(0, 0.01, n_samples)
    
    X = np.column_stack((enrichment, density))
    y = keff
    
    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# ==========================================
# 2. 网页前端布局（Streamlit 核心）
# ==========================================
st.set_page_config(page_title="反应堆参数智能预测系统", page_icon="☢️", layout="wide")

st.title("☢️ 反应堆参数智能预测系统 V1.0")
st.markdown("基于 OpenMC 数据与 AI 代理模型（随机森林/BP神经网络）开发")

# 侧边栏：放置滑块
st.sidebar.header("⚙️ 设计参数调节")
enrichment_input = st.sidebar.slider("U-235 富集度 (%)", min_value=1.5, max_value=5.0, value=3.2, step=0.1)
density_input = st.sidebar.slider("慢化剂密度 (g/cm³)", min_value=0.6, max_value=0.8, value=0.74, step=0.01)

# 核心预测
if st.sidebar.button("🚀 开始预测"):
    # 调用模型
    prediction = model.predict([[enrichment_input, density_input]])[0]
    
    # 判断状态
    if prediction > 1.0:
        status = "超临界 (Supercritical)"
        color = "red"
    elif prediction < 1.0:
        status = "次临界 (Subcritical)"
        color = "blue"
    else:
        status = "临界 (Critical)"
        color = "green"

    # 展示结果面板
    col1, col2, col3 = st.columns(3)
    col1.metric("当前富集度", f"{enrichment_input} %")
    col2.metric("当前密度", f"{density_input} g/cm³")
    col3.metric("AI 预测 k_eff", f"{prediction:.5f}", delta=f"{prediction-1.0:.5f}")
    
    st.markdown(f"### 系统状态：<span style='color:{color}; font-size:24px;'>{status}</span>", unsafe_allow_html=True)

    # ==========================================
    # 3. 画图：富集度对 k_eff 的影响曲线
    # ==========================================
    st.markdown("---")
    st.subheader("📈 预测曲线：富集度 vs k_eff")
    
    # 固定当前密度，让富集度从 1.5 到 5.0 扫一遍
    enr_range = np.linspace(1.5, 5.0, 50)
    den_array = np.full_like(enr_range, density_input)
    
    X_plot = np.column_stack((enr_range, den_array))
    k_eff_preds = model.predict(X_plot)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(enr_range, k_eff_preds, label='Predicted k_eff', color='blue', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Critical (k=1.0)')
    # 标记当前位置
    ax.scatter([enrichment_input], [prediction], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Enrichment (%)')
    ax.set_ylabel('k-effective')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    import time

st.markdown("---")
st.subheader("🔄 物理-热工耦合迭代动态演示")

if st.button("启动耦合迭代演示"):
    # 模拟你的 coupled_pin.py 中的逻辑
    T_current = 600.0
    relax = 0.5
    history_T = []
    history_k = []
    
    chart_placeholder = st.empty() # 用于动态刷新图表
    
    for i in range(1, 10):
        # 模拟物理计算（这里用假公式代替，避免网页卡死）
        power_density = 1.0e8 / (1 + (T_current - 600) * 0.0005) 
        T_new_calc = 600.0 + power_density * 3.0e-6
        
        # 松弛迭代
        T_next = T_current + relax * (T_new_calc - T_current)
        
        # 记录历史
        history_T.append(T_next)
        
        # 动态画图
        chart_placeholder.line_chart(history_T)
        
        time.sleep(0.5) # 制造停顿感，让导师看清过程
        if abs(T_next - T_current) < 1.0:
            st.success(f"迭代收敛！最终燃料温度为: {T_next:.2f} K")
            break
        T_current = T_next