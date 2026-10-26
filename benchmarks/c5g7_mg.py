import openmc

# ==========================================
# 第一步：加载 C5G7 专用的 7 群多群截面库
# ==========================================
mgxs_file = 'c5g7-mgxs.h5'
mg_library = openmc.MGXSLibrary.from_hdf5(mgxs_file)
print("成功加载多群截面库！")

# ==========================================
# 第二步：定义 C5G7 的宏观材料
# ==========================================
# 注意：名称必须与 .h5 文件中严格一致（区分大小写）
# 定义 UO2 材料
uo2 = openmc.Material(name='UO2')
uo2.add_macroscopic('uo2')  # 小写 uo2

# 定义 MOX 材料 (这里选用 mox7 代表 7.0% 富集度的 MOX)
mox = openmc.Material(name='MOX')
mox.add_macroscopic('mox7') # 使用 mox7

# 定义 Water 材料
water = openmc.Material(name='Water')
water.add_macroscopic('water') # 小写 water

materials = openmc.Materials([uo2, mox, water])
materials.cross_sections = 'c5g7-mgxs.h5'
materials.export_to_xml()

# ==========================================
# 第三步：定义简单的 7 群几何（简易版 2x2 阵列）
# ==========================================
# 尺寸参数（C5G7 标准）
pitch = 1.26
pin_r = 0.54

fuel_cyl = openmc.ZCylinder(r=pin_r)

# 定义 UO2 栅元
uo2_cell = openmc.Cell(fill=uo2, region=-fuel_cyl)
mod_cell_uo2 = openmc.Cell(fill=water, region=+fuel_cyl)
uo2_universe = openmc.Universe(cells=[uo2_cell, mod_cell_uo2])

# 定义 MOX 栅元
mox_cell = openmc.Cell(fill=mox, region=-fuel_cyl)
mod_cell_mox = openmc.Cell(fill=water, region=+fuel_cyl)
mox_universe = openmc.Universe(cells=[mox_cell, mod_cell_mox])

# 构建 2x2 阵列（左上/右下 UO2，右上/左下 MOX）
lattice = openmc.RectLattice()
lattice.lower_left = (-pitch, -pitch)
lattice.pitch = (pitch, pitch)
lattice.universes = [
    [uo2_universe, mox_universe],
    [mox_universe, uo2_universe]
]

# 给阵列加边界
x_min = openmc.XPlane(-pitch, boundary_type='reflective')
x_max = openmc.XPlane(pitch, boundary_type='reflective')
y_min = openmc.YPlane(-pitch, boundary_type='reflective')
y_max = openmc.YPlane(pitch, boundary_type='reflective')

assembly_cell = openmc.Cell(fill=lattice, region=+x_min & -x_max & +y_min & -y_max)
geometry = openmc.Geometry([assembly_cell])
geometry.export_to_xml()

# ==========================================
# 第四步：设置多群计算参数
# ==========================================
settings = openmc.Settings()
settings.energy_mode = 'multi-group'
settings.run_mode = 'eigenvalue'
settings.batches = 200
settings.inactive = 50
settings.particles = 10000

source = openmc.IndependentSource()
source.space = openmc.stats.Box((-pitch, -pitch, -10), (pitch, pitch, 10))
settings.source = source
settings.export_to_xml()
# 设置画图（切面图）
plot = openmc.Plot()
plot.filename = 'c5g7_2x2_slice'
plot.basis = 'xy'                # 沿 xy 平面剖开
plot.origin = (0., 0., 0.)       # 切面中心点
plot.width = (2.52, 2.52)        # 画布大小 (pitch 1.26 * 2)
plot.pixels = (500, 500)         # 分辨率
plot.color_by = 'material'       # 按材料上色
plots = openmc.Plots([plot])
plots.export_to_xml()
print("C5G7 多群截面 2x2 模型输入卡生成完毕！")
