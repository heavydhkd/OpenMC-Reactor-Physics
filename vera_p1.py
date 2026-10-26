import openmc
import os

# 每次运行前清理旧文件，防止数据冲突
def clean_previous_files():
    files_to_remove = ['materials.xml', 'geometry.xml', 'settings.xml', 
                       'summary.h5', 'statepoint.100.h5', 'tallies.xml']
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)

def run_vera_p1(fuel_temp_K):
    clean_previous_files()
    
    # ==========================================
    # 1. 定义材料（严格遵循 VERA P1 参数）
    # ==========================================
    fuel = openmc.Material(name='UO2')
    fuel.add_element('U', 1, enrichment=4.95) # 4.95% 富集度
    fuel.add_element('O', 2)
    fuel.set_density('g/cm3', 10.257)
    fuel.temperature = fuel_temp_K # 核心：设置燃料温度

    # 补充：包壳材料 Zircaloy
    clad = openmc.Material(name='Zircaloy')
    clad.add_element('Zr', 1)
    clad.set_density('g/cm3', 6.56)

    water = openmc.Material(name='Water')
    water.add_element('H', 2)
    water.add_element('O', 1)
    water.set_density('g/cm3', 0.743)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = 600.0 # 慢化剂温度固定 600K

    mats = openmc.Materials([fuel, clad, water])
    mats.export_to_xml()

    # ==========================================
    # 2. 定义几何（补齐包壳）
    # ==========================================
    fuel_cyl = openmc.ZCylinder(r=0.4096)
    clad_cyl = openmc.ZCylinder(r=0.475)
    
    # 燃料区：圆柱内部
    fuel_cell = openmc.Cell(fill=fuel, region=-fuel_cyl)
    # 包壳区：燃料外径到包壳外径之间（这就是你之前漏掉的区域！）
    clad_cell = openmc.Cell(fill=clad, region=+fuel_cyl & -clad_cyl)
    # 慢化剂区：包壳外径以外，直到外部边界
    mod_cell = openmc.Cell(fill=water, region=+clad_cyl)
    
    # 定义全反射边界（正方形，边长 1.26 cm）
    x_min = openmc.XPlane(-0.63, boundary_type='reflective')
    x_max = openmc.XPlane(0.63, boundary_type='reflective')
    y_min = openmc.YPlane(-0.63, boundary_type='reflective')
    y_max = openmc.YPlane(0.63, boundary_type='reflective')

    # 把所有栅元塞进一个正方形宇宙
    root = openmc.Cell(fill=openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]), 
                       region=+x_min & -x_max & +y_min & -y_max)
    geometry = openmc.Geometry([root])
    geometry.export_to_xml()

    # ==========================================
    # 3. 计算设置
    # ==========================================
    settings = openmc.Settings()
    settings.run_mode = 'eigenvalue'
    settings.batches = 100
    settings.inactive = 20
    settings.particles = 5000
    
    # 点源放在原点
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0,0,0))
    settings.source = source
    settings.export_to_xml()

    # ==========================================
    # 4. 运行并读取结果
    # ==========================================
    # 注意：这里默认 output=False，如果还有报错，可以改成 True 看详细日志
    openmc.run(output=False)
    
    sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
    # 提取 k_eff
    keff = sp.keff.nominal_value
    print(f"\n>>> 燃料温度 {fuel_temp_K} K -> k_eff = {keff:.5f} <<<\n")

# 跑两个子算例：600K 和 900K
run_vera_p1(600.0)
run_vera_p1(900.0)