import openmc
import os

# 循环计算 1.6%、2.4%、3.2%、4.95% 四个富集度
for enrichment in [1.6, 2.4, 3.2, 4.95]:
    print(f"\n正在计算富集度: {enrichment}% ...")
    
    # ==========================================
    # 关键修复：每次循环前，清理上一次生成的残留文件
    for f in os.listdir('.'):
        if f.endswith('.xml') or f.endswith('.h5'):
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
    
    mats = openmc.Materials([fuel, water])
    mats.export_to_xml()
    
    # 2. 定义几何
    fuel_cyl = openmc.ZCylinder(r=0.41)
    clad_cyl = openmc.ZCylinder(r=0.475)
    fuel_cell = openmc.Cell(name='Fuel', fill=fuel, region=-fuel_cyl)
    clad_cell = openmc.Cell(name='Clad', fill=water, region=+fuel_cyl & -clad_cyl)
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
    
    # 关闭文件句柄，避免下一次循环冲突
    sp.close()

print("\n扫描完成！")
