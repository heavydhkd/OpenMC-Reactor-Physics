import openmc
import os
import csv

# 扫描结果（最后写入 benchmarks/scan_results.csv，供 plot.py 读取）
results = []

# 循环计算 1.6%、2.4%、3.2%、4.95% 四个富集度
for enrichment in [1.6, 2.4, 3.2, 4.95]:
    print(f"\n正在计算富集度: {enrichment}% ...")
    
    # ==========================================
    # 关键修复：每次循环前只清理上一次生成的 statepoint 文件
    # （避免误删 C5G7 多群截面库 c5g7-mgxs.h5）
    for f in os.listdir('.'):
        if f.startswith('statepoint.') and f.endswith('.h5'):
            os.remove(f)
    # ==========================================
    
    # 1. 定义材料
    fuel = openmc.Material(name='UO2')
    fuel.add_element('U', 1, enrichment=enrichment)
    fuel.add_element('O', 2)
    fuel.set_density('g/cm3', 10.4)
    
    water = openmc.Material(name='Water')
    water.add_element('H', 2)
    water.add_element('O', 1)
    water.set_density('g/cm3', 0.74)
    water.add_s_alpha_beta('c_H_in_H2O')
    
    # 包壳材料：Zircaloy（Zr + 少量 Sn/Fe/Cr/O）
    clad = openmc.Material(name='Zircaloy')
    clad.add_element('Zr', 0.9828)
    clad.add_element('Sn', 0.013)
    clad.add_element('Fe', 0.002)
    clad.add_element('Cr', 0.001)
    clad.add_element('O', 0.0012)
    clad.set_density('g/cm3', 6.56)

    mats = openmc.Materials([fuel, clad, water])
    mats.export_to_xml()
    
    # 2. 定义几何
    fuel_cyl = openmc.ZCylinder(r=0.41)
    clad_cyl = openmc.ZCylinder(r=0.475)
    fuel_cell = openmc.Cell(name='Fuel', fill=fuel, region=-fuel_cyl)
    clad_cell = openmc.Cell(name='Clad', fill=clad, region=+fuel_cyl & -clad_cyl)
    mod_cell = openmc.Cell(name='Moderator', fill=water, region=+clad_cyl)
    
    x_min = openmc.XPlane(-0.63, boundary_type='reflective')
    x_max = openmc.XPlane(0.63, boundary_type='reflective')
    y_min = openmc.YPlane(-0.63, boundary_type='reflective')
    y_max = openmc.YPlane(0.63, boundary_type='reflective')
    
    root_cell = openmc.Cell(name='Root', fill=openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]),
                            region=+x_min & -x_max & +y_min & -y_max)
    geometry = openmc.Geometry([root_cell])
    geometry.export_to_xml()
    
    # 3. 设置计算参数
    settings = openmc.Settings()
    settings.run_mode = 'eigenvalue'
    settings.batches = 50
    settings.inactive = 10
    settings.particles = 2000
    
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0,0,0))
    settings.source = source
    settings.export_to_xml()
    
    # 4. 运行并读取结果
    openmc.run(output=False)
    sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
    keff = sp.keff
    print(f"富集度 {enrichment}% -> k_eff = {keff.nominal_value:.5f} ± {keff.std_dev:.5f}")
    results.append({'enrichment': enrichment, 'k_eff': keff.nominal_value, 'std': keff.std_dev})
    
    # 关闭文件句柄，避免下一次循环冲突
    sp.close()

# 5. 将扫描结果写入 CSV（固定写到脚本所在的 benchmarks/ 目录）
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scan_results.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['enrichment', 'k_eff', 'std'])
    writer.writeheader()
    writer.writerows(results)

print(f"\n扫描完成！结果已写入 {csv_path}")
