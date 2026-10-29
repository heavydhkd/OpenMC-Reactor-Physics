# benchmarks：传统基准题脚本

本目录存放项目早期的**独立基准题脚本**，用于手动生成 OpenMC 输入卡、运行单个
算例或做简单对比。它们不属于 `src/` 中的代理模型与逆向寻优主流程，也**没有**
接入 CI 或单元测试，改动后需要自己运行验证。

## 运行前提

- 需要已安装可用的 OpenMC（建议 0.13+），且 `import openmc` 能正常工作；
- 多群算例还需要 C5G7 多群截面库 `c5g7-mgxs.h5`。该文件**不随 OpenMC 分发**，
  必须另行获取并放到运行目录，否则 `c5g7_core.py` 和 `c5g7_mg.py` 无法启动；
- 脚本会在当前工作目录生成 `materials.xml`、`geometry.xml`、`settings.xml`、
  `plots.xml`、`statepoint.*.h5` 等文件，请勿在存放重要 HDF5 文件的目录运行；
- `scan.py` 只清理 `statepoint.*.h5`，不会删除截面库等其他 `.h5` 文件。

## 脚本一览

| 脚本 | 作用 |
| --- | --- |
| `build_model.py` | 生成 3.2% 富集度 UO2 单栅元（含 Zircaloy 包壳）输入卡 |
| `scan.py` | 扫描 1.6%、2.4%、3.2%、4.95% 四个富集度，结果写入 `scan_results.csv` |
| `plot.py` | 读取 `scan_results.csv`，绘制 k_eff 随富集度变化曲线 |
| `assembly.py` | 生成 2x2 单栅元阵列（全反射边界）输入卡 |
| `c5g7_core.py` | 简化版 C5G7 2x2 组件堆芯（多群），生成输入卡与切片图 |
| `c5g7_mg.py` | 更简化的 C5G7 2x2 阵列多群模型，导出 `c5g7_2x2_slice.png` |
| `vera_p1.py` | 计算 VERA P1 单栅元在 600 K / 900 K 燃料温度下的 k_eff |

扫描与绘图顺序：先运行 `python benchmarks/scan.py`，再运行
`python benchmarks/plot.py`；缺少 CSV 时 `plot.py` 会提示先执行扫描。

## 关于 C5G7 结果的说明

`c5g7_core.py` 只是**简化**的 2x2 组件模型，并不是官方 C5G7 2x2 堆芯布局：

- 导向管被排成贯穿组件的整条十字，官方布局中它们只出现在特定位置；
- 裂变室填满了整个 1.26 cm 栅元，官方模型中只是半径 0.54 cm 的圆；
- 燃料/慢化剂等分区也做了简化。

因此该脚本算出的 k_eff 只能说明代码能跑通，**不能**与已发表的 C5G7 基准题
参考解做定量比较；`c5g7_mg.py` 同样如此。仓库中的切片图仅作示意，不代表
官方基准结果。
