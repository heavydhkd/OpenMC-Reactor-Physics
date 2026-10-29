import openmc

# 1. 定义材料（3.2%富集度 UO2 燃料 + 轻水慢化剂）
fuel = openmc.Material(name='UO2')
fuel.add_element('U', 1, enrichment=3.2)
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

# 2. 定义几何（燃料芯块、包壳、水）
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
settings.batches = 100
settings.inactive = 20
settings.particles = 5000

source = openmc.IndependentSource()
source.space = openmc.stats.Point((0,0,0))
settings.source = source
settings.export_to_xml()

print("官方标准单栅元输入文件生成成功！")
