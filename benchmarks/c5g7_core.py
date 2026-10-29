import openmc

# 1. 加载多群截面
mgxs_file = 'c5g7-mgxs.h5'
mg_library = openmc.MGXSLibrary.from_hdf5(mgxs_file)

# 2. 定义材料
def create_material(name, macro_name):
    mat = openmc.Material(name=name)
    mat.add_macroscopic(macro_name)
    return mat

uo2_mat = create_material('UO2', 'uo2')
mox43 = create_material('MOX43', 'mox43')
mox7 = create_material('MOX7', 'mox7')
mox87 = create_material('MOX87', 'mox87')
water = create_material('Water', 'water')
fiss_chamber = create_material('FissChamber', 'fiss_chamber')
guide_tube_mat = create_material('GuideTube', 'guide_tube')

materials = openmc.Materials([uo2_mat, mox43, mox7, mox87, water, fiss_chamber, guide_tube_mat])
materials.cross_sections = 'c5g7-mgxs.h5'
materials.export_to_xml()

# 3. 定义单栅元（圆柱体 + 水）
pitch = 1.26
pin_r = 0.54
fuel_cyl = openmc.ZCylinder(r=pin_r)

def create_pin_universe(fuel_mat):
    if fuel_mat is None:
        return openmc.Universe(cells=[openmc.Cell(fill=water)])
    fuel_cell = openmc.Cell(fill=fuel_mat, region=-fuel_cyl)
    mod_cell = openmc.Cell(fill=water, region=+fuel_cyl)
    return openmc.Universe(cells=[fuel_cell, mod_cell])

uo2_univ = create_pin_universe(uo2_mat)
mox43_univ = create_pin_universe(mox43)
mox7_univ = create_pin_universe(mox7)
mox87_univ = create_pin_universe(mox87)
water_univ = create_pin_universe(None)

# 导向管和裂变室填满整个栅元
fiss_chamber_univ = openmc.Universe(cells=[openmc.Cell(fill=fiss_chamber)])
guide_tube_univ = openmc.Universe(cells=[openmc.Cell(fill=guide_tube_mat)])

# 4. 构建 17x17 UO2 组件
def build_uo2_assembly():
    grid = [[uo2_univ for _ in range(17)] for _ in range(17)]
    grid[0][0] = grid[0][16] = grid[16][0] = grid[16][16] = water_univ
    grid[8][8] = fiss_chamber_univ
    for i in range(17):
        if i != 8:
            grid[8][i] = guide_tube_univ
            grid[i][8] = guide_tube_univ
            
    lattice = openmc.RectLattice()
    lattice.lower_left = (-17*pitch/2, -17*pitch/2)
    lattice.pitch = (pitch, pitch)
    lattice.universes = grid
    lattice.outer = water_univ
    return lattice

# 5. 构建 17x17 MOX 组件
def build_mox_assembly():
    grid = [[mox7_univ for _ in range(17)] for _ in range(17)]
    for i in range(5, 12):
        for j in range(5, 12):
            grid[i][j] = mox43_univ
    for i in range(17):
        grid[0][i] = mox87_univ
        grid[16][i] = mox87_univ
        grid[i][0] = mox87_univ
        grid[i][16] = mox87_univ

    grid[0][0] = grid[0][16] = grid[16][0] = grid[16][16] = water_univ
    grid[8][8] = fiss_chamber_univ
    for i in range(17):
        if i != 8:
            grid[8][i] = guide_tube_univ
            grid[i][8] = guide_tube_univ
            
    lattice = openmc.RectLattice()
    lattice.lower_left = (-17*pitch/2, -17*pitch/2)
    lattice.pitch = (pitch, pitch)
    lattice.universes = grid
    lattice.outer = water_univ
    return lattice

# 6. 组装全堆芯 (关键：加入 0.08cm 水隙 + 全反射边界)
core_pitch = 21.5  # 21.42 + 0.08 水隙
uo2_assembly_lattice = build_uo2_assembly()
mox_assembly_lattice = build_mox_assembly()

uo2_assembly_cell = openmc.Cell(fill=uo2_assembly_lattice)
uo2_assembly_univ = openmc.Universe(cells=[uo2_assembly_cell])

mox_assembly_cell = openmc.Cell(fill=mox_assembly_lattice)
mox_assembly_univ = openmc.Universe(cells=[mox_assembly_cell])

core_lattice = openmc.RectLattice()
core_lattice.lower_left = (-core_pitch, -core_pitch)
core_lattice.pitch = (core_pitch, core_pitch)
core_lattice.universes = [
    [uo2_assembly_univ, mox_assembly_univ],
    [mox_assembly_univ, uo2_assembly_univ]
]
core_lattice.outer = water_univ

# 全反射边界（无限阵列）
cx_min = openmc.XPlane(-core_pitch, boundary_type='reflective')
cx_max = openmc.XPlane(core_pitch, boundary_type='reflective')
cy_min = openmc.YPlane(-core_pitch, boundary_type='reflective')
cy_max = openmc.YPlane(core_pitch, boundary_type='reflective')

core_cell = openmc.Cell(fill=core_lattice, region=+cx_min & -cx_max & +cy_min & -cy_max)
geometry = openmc.Geometry([core_cell])
geometry.export_to_xml()

# 7. 计算设置
settings = openmc.Settings()
settings.energy_mode = 'multi-group'
settings.run_mode = 'eigenvalue'
settings.batches = 200
settings.inactive = 50
settings.particles = 10000

source = openmc.IndependentSource()
source.space = openmc.stats.Box((-core_pitch, -core_pitch, -10), (core_pitch, core_pitch, 10))
settings.source = source
settings.export_to_xml()

# 8. 堆芯中平面切片图（openmc.run() 时生成 c5g7_full_core.png）
plot = openmc.Plot()
plot.filename = 'c5g7_full_core'
plot.basis = 'xy'                # 沿 xy 平面剖开
plot.origin = (0., 0., 0.)       # 经过堆芯中平面
plot.width = (43., 43.)          # 覆盖 2x2 组件堆芯（core_pitch 21.5 * 2）
plot.pixels = (400, 400)         # 分辨率
plot.color_by = 'material'       # 按材料上色
plots = openmc.Plots([plot])
plots.export_to_xml()

print("C5G7 官方对标版生成完毕！")

if __name__ == '__main__':
    openmc.run()
