import openmc

# 1. 加载多群截面
mgxs_file = 'c5g7-mgxs.h5'
mg_library = openmc.MGXSLibrary.from_hdf5(mgxs_file)

# 2. 定义材料
def create_material(name, macro_name):
    mat = openmc.Material(name=name)
    mat.add_macroscopic(macro_name)
    return mat

uo2 = create_material('UO2', 'uo2')
mox = create_material('MOX', 'mox7') # 简化处理，统一用mox7
water = create_material('Water', 'water')

materials = openmc.Materials([uo2, mox, water])
materials.cross_sections = 'c5g7-mgxs.h5'
materials.export_to_xml()

# 3. 基础栅元 Universe
pitch = 1.26
pin_r = 0.54
fuel_cyl = openmc.ZCylinder(r=pin_r)

def create_pin_universe(fuel_mat):
    fuel_cell = openmc.Cell(fill=fuel_mat, region=-fuel_cyl)
    mod_cell = openmc.Cell(fill=water, region=+fuel_cyl)
    return openmc.Universe(cells=[fuel_cell, mod_cell])

uo2_univ = create_pin_universe(uo2)
mox_univ = create_pin_universe(mox)
water_univ = openmc.Universe(cells=[openmc.Cell(fill=water)]) # 纯水孔

# 4. 构建 17x17 组件生成函数
def build_assembly(base_fuel_univ):
    # 17x17 全填燃料
    grid = [[base_fuel_univ for _ in range(17)] for _ in range(17)]
    
    # 四个角是纯水孔
    grid[0][0] = water_univ
    grid[0][16] = water_univ
    grid[16][0] = water_univ
    grid[16][16] = water_univ
    
    # 真实的C5G7有特定的控制棒导向管位置，这里我们做个十字形纯水简化
    for i in range(5, 12):
        grid[8][i] = water_univ  # 横排
        grid[i][8] = water_univ  # 竖排
        
    lattice = openmc.RectLattice()
    lattice.lower_left = (-17*pitch/2, -17*pitch/2)
    lattice.pitch = (pitch, pitch)
    lattice.universes = grid
    
    # 组件最外层边界（水隙边界）
    x_min = openmc.XPlane(-17*pitch/2, boundary_type='reflective')
    x_max = openmc.XPlane(17*pitch/2, boundary_type='reflective')
    y_min = openmc.YPlane(-17*pitch/2, boundary_type='reflective')
    y_max = openmc.YPlane(17*pitch/2, boundary_type='reflective')
    
    cell = openmc.Cell(fill=lattice, region=+x_min & -x_max & +y_min & -y_max)
    return openmc.Universe(cells=[cell])

    # 5. 生成两个不同的组件 Universe
uo2_assembly = build_assembly(uo2_univ)
mox_assembly = build_assembly(mox_univ)

# 6. 拼装 2x2 宏观堆芯 (右上/左下是MOX，左上/右下是UO2)
core_lattice = openmc.RectLattice()
core_pitch = 17 * pitch  # 组件边长
core_lattice.lower_left = (-core_pitch, -core_pitch)
core_lattice.pitch = (core_pitch, core_pitch)
core_lattice.universes = [
    [uo2_assembly, mox_assembly],
    [mox_assembly, uo2_assembly]
]

# 堆芯边界 (反射边界)
cx_min = openmc.XPlane(-core_pitch, boundary_type='reflective')
cx_max = openmc.XPlane(core_pitch, boundary_type='reflective')
cy_min = openmc.YPlane(-core_pitch, boundary_type='reflective')
cy_max = openmc.YPlane(core_pitch, boundary_type='reflective')

core_cell = openmc.Cell(fill=core_lattice, region=+cx_min & -cx_max & +cy_min & -cy_max)
geometry = openmc.Geometry([core_cell])
geometry.export_to_xml()

# 7. 设置计算参数 (全尺寸计算需要更多粒子)
settings = openmc.Settings()
settings.energy_mode = 'multi-group'
settings.run_mode = 'eigenvalue'
settings.batches = 200
settings.inactive = 50
settings.particles = 10000

# 源分布设为覆盖整个堆芯
source = openmc.IndependentSource()
source.space = openmc.stats.Box((-core_pitch, -core_pitch, -10), (core_pitch, core_pitch, 10))
settings.source = source
settings.export_to_xml()

# 8. 画图：看看真正的全堆芯！
plot = openmc.Plot()
plot.filename = 'c5g7_full_core'
plot.basis = 'xy'
plot.origin = (0., 0., 0.)
plot.width = (core_pitch*2, core_pitch*2)
plot.pixels = (1000, 1000)  # 高清分辨率
plot.color_by = 'material'
plots = openmc.Plots([plot])
plots.export_to_xml()

print("完整 C5G7 2D 堆芯输入卡生成完毕！")