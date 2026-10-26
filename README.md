#  OpenMC-Reactor-Physics: AI 代理模型与智能逆向寻优系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenMC](https://img.shields.io/badge/OpenMC-0.13%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellowgreen)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 项目简介

在核反应堆物理计算中，高保真蒙特卡洛（Monte Carlo）程序（如 OpenMC）虽然计算精度极高，但其单次运算耗时较长，难以直接应用于需要海量样本（如几十万组）的多目标优化、敏感性分析或数字孪生中。

本项目以 **VERA Benchmark Problem 1 (PWR 单栅元)** 为物理对象，旨在构建一套**“高保真数据生成 -> 机器学习代理模型 (Surrogate Model) -> 智能逆向寻优 -> 主动学习 (Active Learning)”**的完整闭环系统。

### 核心亮点
-  **面向对象物理建模**：将复杂的物理几何、材料、参数封装为 `VeraPinCell` 类，一行代码驱动模型。
-  **高效数据生成引擎**：结合拉丁超立方采样 (LHS) 与 Python 多进程并发 (`ProcessPoolExecutor`)，10 分钟内并行生成 100 组高保真数据集。
-  **高精度代理模型**：引入高斯过程回归 (Kriging) 完美处理小样本与蒙特卡洛统计涨落噪声，测试集误差低至 **~134 pcm**，\(R^2\) 达 **0.9996**。
-  **智能逆向设计**：利用 `scipy.optimize` 驱动代理模型，在毫秒级时间内完成临界状态的参数寻优，并自动调用 OpenMC 进行真实复核。
-  **主动学习闭环**：将寻优结果自动回流至数据库更新模型，实现 AI 的自我迭代。

---

## 系统架构与技术路线

```mermaid
graph TD
    A[configs/params.yaml 定义物理边界] --> B(拉丁超立方 LHS 采样)
    B --> C{多进程并行 OpenMC 跑批}
    C --> D[data/processed/vera_p1_dataset.csv 高保真数据集]
    
    D --> E[训练代理模型]
    E --> F1(Kriging / 高斯过程回归)
    E --> F2(BP 神经网络)
    
    F1 --> G{智能逆向寻优 SLSQP}
    G --> |目标: k_eff = 1.0| H[候选最优参数组合]
    H --> I[真实 OpenMC 高保真验证]
    I --> J[主动学习: 将新数据回流至数据集]
    J --> E
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style F1 fill:#ffb,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px