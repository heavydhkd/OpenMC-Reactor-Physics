import openmc

# 1. 定义材料
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

# 2. 几何尺寸参数
pitch = 1.26
pin_r = 0.41
clad_r = 0.475

fuel_cyl = openmc.ZCylinder(r=pin_r)
clad_cyl = openmc.ZCylinder(r=clad_r)
fuel_cell = openmc.Cell(fill=fuel, region=-fuel_cyl)
clad_cell = openmc.Cell(fill=clad, region=+fuel_cyl & -clad_cyl)
mod_cell = openmc.Cell(fill=water, region=+clad_cyl)

x_min = openmc.XPlane(-pitch/2, boundary_type='reflective')
x_max = openmc.XPlane(pitch/2, boundary_type='reflective')
y_min = openmc.YPlane(-pitch/2, boundary_type='reflective')
y_max = openmc.YPlane(pitch/2, boundary_type='reflective')

pin_universe = openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell])
root_cell = openmc.Cell(fill=pin_universe, region=+x_min & -x_max & +y_min & -y_max)
base_universe = openmc.Universe(cells=[root_cell])

# 3. 构建 2x2 阵列
lattice = openmc.RectLattice()
lattice.lower_left = (-pitch, -pitch)
lattice.pitch = (pitch, pitch)
lattice.universes = [
    [base_universe, base_universe],
    [base_universe, base_universe]
]

assembly_x_min = openmc.XPlane(-pitch, boundary_type='reflective')
assembly_x_max = openmc.XPlane(pitch, boundary_type='reflective')
assembly_y_min = openmc.YPlane(-pitch, boundary_type='reflective')
assembly_y_max = openmc.YPlane(pitch, boundary_type='reflective')

assembly_cell = openmc.Cell(fill=lattice, region=+assembly_x_min & -assembly_x_max & +assembly_y_min & -assembly_y_max)
geometry = openmc.Geometry([assembly_cell])
geometry.export_to_xml()

# 4. 设置计算参数
settings = openmc.Settings()
settings.run_mode = 'eigenvalue'
settings.batches = 100
settings.inactive = 20
settings.particles = 10000

source = openmc.IndependentSource()
source.space = openmc.stats.Box((-pitch, -pitch, -10), (pitch, pitch, 10))
settings.source = source
settings.export_to_xml()
print("2x2 组件输入文件生成完毕！")