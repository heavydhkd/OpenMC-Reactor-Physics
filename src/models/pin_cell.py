import openmc
import os
import h5py
import glob

class VeraPinCell:
    def __init__(self, enrichment=4.5, boron_ppm=500.0, clad_outer_r=0.4750,
                 fuel_temp=900.0, mod_temp=600.0):
        """
        参数化 VERA P1 栅元
        :param enrichment: U235 富集度 (%)
        :param boron_ppm: 冷却剂中的硼浓度 (ppm)
        :param clad_outer_r: 包壳外半径 (cm)
        :param fuel_temp: 燃料温度 (K)
        :param mod_temp: 慢化剂温度 (K)
        """
        self.enrichment = enrichment
        self.boron_ppm = boron_ppm
        self.clad_outer_r = clad_outer_r
        
        # 假设包壳厚度恒为 0.057 cm，自动计算内半径
        self.clad_inner_r = clad_outer_r - 0.057 
        
        self.fuel_temp = fuel_temp
        self.mod_temp = mod_temp

    def _build_materials(self):
        # 1. 燃料 (UO2)
        fuel = openmc.Material(name='UO2')
        fuel.add_element('U', 1.0, enrichment=self.enrichment)
        fuel.add_element('O', 2.0)
        fuel.set_density('g/cm3', 10.257)
        fuel.temperature = self.fuel_temp  # 动态传入燃料温度

        # 2. 包壳 (Zircaloy-4)
        clad = openmc.Material(name='Zircaloy')
        clad.add_element('Zr', 0.98)
        clad.add_element('Sn', 0.015)
        clad.set_density('g/cm3', 6.56)
        clad.temperature = self.mod_temp  # 包壳温度通常近似为慢化剂温度

        # 3. 冷却剂 (含硼水)
        water = openmc.Material(name='Water')
        water.add_element('H', 2.0)
        water.add_element('O', 1.0)
        water.set_density('g/cm3', 0.743)
        water.add_s_alpha_beta('c_H_in_H2O')
        water.temperature = self.mod_temp  # 动态传入慢化剂温度
        
        if self.boron_ppm > 0:
            water.add_element('B', self.boron_ppm * 1e-6)

        self.materials = openmc.Materials([fuel, clad, water])

    def _build_geometry(self):
        # 构建圆柱面
        fuel_cyl = openmc.ZCylinder(r=self.clad_inner_r)
        clad_cyl = openmc.ZCylinder(r=self.clad_outer_r)
        
        # 构建 Cell
        fuel_cell = openmc.Cell(name='Fuel', fill=self.materials[0], region=-fuel_cyl)
        clad_cell = openmc.Cell(name='Clad', fill=self.materials[1], region=+fuel_cyl & -clad_cyl)
        
        pitch = 1.26
        half_pitch = pitch / 2.0
        mod_cell = openmc.Cell(name='Moderator', fill=self.materials[2], region=+clad_cyl)

        # 边界 (全反射)
        x_min = openmc.XPlane(-half_pitch, boundary_type='reflective')
        x_max = openmc.XPlane(half_pitch, boundary_type='reflective')
        y_min = openmc.YPlane(-half_pitch, boundary_type='reflective')
        y_max = openmc.YPlane(half_pitch, boundary_type='reflective')

        root_cell = openmc.Cell(name='Root',
            fill=openmc.Universe(cells=[fuel_cell, clad_cell, mod_cell]),
            region=+x_min & -x_max & +y_min & -y_max)
        
        self.geometry = openmc.Geometry([root_cell])

    def _build_settings(self):
        self.settings = openmc.Settings()
        self.settings.run_mode = 'eigenvalue'
        self.settings.batches = 100
        self.settings.inactive = 20
        self.settings.particles = 10000
        
        # 点源 (0,0,0)
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        self.settings.source = source

    def run_and_extract(self, output_dir: str):
        """构建模型，运行 OpenMC，提取 k_eff 和标准差"""
        self._build_materials()
        self._build_geometry()
        self._build_settings()
        
        model = openmc.model.Model(
            geometry=self.geometry,
            materials=self.materials,
            settings=self.settings
        )
        
        os.makedirs(output_dir, exist_ok=True)
        out_dir_abs = os.path.abspath(output_dir)
        
        # 运行 OpenMC，仅使用 cwd 参数
        model.run(cwd=out_dir_abs)
        
        # 动态寻找最新生成的 statepoint 文件
        sp_files = glob.glob(os.path.join(out_dir_abs, 'statepoint.*.h5'))
        if not sp_files:
            raise FileNotFoundError(f"在 {out_dir_abs} 下找不到 statepoint 文件！")
        sp_path = max(sp_files, key=os.path.getctime)
        
        # 用 h5py 直接读底层数据
        with h5py.File(sp_path, 'r') as f:
            k_combined = f['k_combined'][()]
            k_eff = float(k_combined[0])
            k_std = float(k_combined[1])
            
        return {'k_eff': k_eff, 'std': k_std}