import openmc
import os

# ==========================================
# 0. 文件清理函数（防止读取上一次的旧数据）
# ==========================================
def clean_files():
    files = ['materials.xml', 'geometry.xml', 'settings.xml', 'tallies.xml', 
             'summary.h5', 'statepoint.50.h5']
    for f in files:
        if os.path.exists(f):
            os.remove(f)

# ==========================================
# 1. 极简一维热工模型 (纯物理公式计算)
# ==========================================
def calculate_fuel_temp(power_density):
    """
    根据体积释热率计算燃料中心温度。
    T_fuel = T_coolant + q''' * R_total
    """
    T_coolant = 600.0  # 冷却剂温度固定为 600K
    # 假设一个简化的综合热阻 (包含气隙、包壳导热、对流换热)
    # 这个数值是人为设定的，目的是让温度变化明显，便于观察收敛
    R_total = 3.0e-6
    
    T_fuel_new = T_coolant + (power_density * R_total)
    return T_fuel_new

# ==========================================
# 2. 单次 OpenMC 物理计算模型
# ==========================================
def run_openmc(fuel_temp_K):
    clean_files()
    
    # --- 材料定义 ---
    fuel = openmc.Material(name='UO2')
    fuel.add_element('U', 1, enrichment=4.95)
    fuel.add_element('O', 2)
    fuel.set_density('g/cm3', 10.257)
    fuel.temperature = fuel_temp_K  # 核心：输入热工反馈的温度

    clad = openmc.Material(name='Zircaloy') # 包壳材料（修复报错的关键）
    clad.add_element('Zr', 1)
    clad.set_density('g/cm3', 6.56)

    water = openmc.Material(name='Water')
    water.add_element('H', 2)
    water.add_element('O', 1)
    water.set_density('g/cm3', 0.743)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = 600.0

    mats = openmc.Materials([fuel, clad, water])
    mats.export_to_xml()

    # --- 几何定义 ---
    fuel_cyl = openmc.ZCylinder(r=0.4096)
    clad_cyl = openmc.ZCylinder(r=0.475)
    
    # 三个区域：燃料内部、包壳层、外部慢化剂
    fuel_cell = openmc.Cell(fill=fuel, region=-fuel_cyl)
    clad_cell = openmc.Cell(fill=clad, region=+fuel_cyl & -clad_cyl) # 修复报错的关键
    mod_cell = openmc.Cell(fill=water, region=+clad_cyl)
    
    # 外部边界：全反射模拟无限阵列
    x_min = openmc.XPlane(-0.63, boundary_type='reflective')
    x_max = openmc.XPlane(0.63, boundary_type='reflective')
    y_min = openmc.YPlane(-0.63, boundary_type='reflective')
    y_max = openmc.YPlane(0.63, boundary_type='reflective')

    root = openmc.Cell(fill=openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]), 
                       region=+x_min & -x_max & +y_min & -y_max)
    geometry = openmc.Geometry([root])
    geometry.export_to_xml()

    # --- Tally 设置 (提取燃料区的能量沉积) ---
    heating_tally = openmc.Tally(name='heating')
    heating_tally.filters = [openmc.CellFilter(fuel_cell)]
    heating_tally.scores = ['heating']
    tallies = openmc.Tallies([heating_tally])
    tallies.export_to_xml()

    # --- 计算设置 ---
    settings = openmc.Settings()
    settings.temperature = {'method': 'interpolation'}
    settings.run_mode = 'eigenvalue'
    settings.batches = 50
    settings.inactive = 10
    settings.particles = 5000
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0,0,0))
    settings.source = source
    settings.export_to_xml()

    # --- 运行并提取数据 ---
    openmc.run(output=False)
    
    sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
    keff = sp.keff.nominal_value
    
    # 提取 heating tally 的值
    tally_res = sp.get_tally(name='heating')
    power_density = tally_res.mean.flatten()[0] 
    
    return keff, power_density

# ==========================================
# 3. 核心：松弛迭代耦合大循环
# ==========================================
if __name__ == "__main__":
    print("=== 开始物理-热工耦合迭代 ===")

    T_current = 600.0  # 初始猜测燃料温度为 600K
    relax_factor = 0.5 # 松弛因子

    for iteration in range(1, 12): 
        print(f"\n[第 {iteration} 次迭代] 传入 OpenMC 的燃料温度: {T_current:.2f} K")
        
        # 1. 跑物理，拿当前温度下的 k_eff 和 功率
        keff, power = run_openmc(T_current)
        
        # 2. 跑热工，算新温度
        T_new_calc = calculate_fuel_temp(power)
        print(f"  -> OpenMC 算出的 k_eff: {keff:.5f}, 相对功率: {power:.2f}")
        print(f"  -> 物理算出的新温度 T_new: {T_new_calc:.2f} K")
        
        # 3. 应用松弛因子 (防止温度剧烈跳跃发散)
        T_next = T_current + relax_factor * (T_new_calc - T_current)
        
        # 4. 判断收敛 (温度变化小于 0.1K 就认为稳定了)
        diff = abs(T_next - T_current)
        if diff < 1.0:
            print(f"\n✅ 达到收敛！最终稳态燃料温度为: {T_next:.2f} K, 最终 k_eff = {keff:.5f}")
            break
            
        T_current = T_next