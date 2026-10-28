import openmc
import os
import h5py
import glob

class VeraPinCell:
    def __init__(self, enrichment=4.5, boron_ppm=500.0, clad_outer_r=0.4750,
                 fuel_temp=900.0, mod_temp=600.0):
        """
        参数化 VERA P1 栅元
        """
        self.enrichment = enrichment
        self.boron_ppm = boron_ppm
        self.clad_outer_r = clad_outer_r
        self.clad_inner_r = clad_outer_r - 0.057 
        self.fuel_temp = fuel_temp
        self.mod_temp = mod_temp

    def _build_materials(self):
        # 1. 燃料 (UO2)
        fuel = openmc.Material(name='UO2')
        fuel.add_element('U', 1.0, enrichment=self.enrichment)
        fuel.add_element('O', 2.0)
        fuel.set_density('g/cm3', 10.257)
        fuel.temperature = self.fuel_temp

        # 2. 包壳 (Zircaloy-4)
        clad = openmc.Material(name='Zircaloy')
        clad.add_element('Zr', 0.98)
        clad.add_element('Sn', 0.015)
        clad.set_density('g/cm3', 6.56)
        clad.temperature = self.mod_temp

        # 3. 冷却剂 (含硼水)
        water = openmc.Material(name='Water')
        water.add_element('H', 2.0)
        water.add_element('O', 1.0)
        
        # 🔑 核心物理修复：水密度随温度动态变化 (简化热膨胀模型)
        # 基准点：600K 对应密度 0.743 g/cm3。
        # 压水堆中，温度升高 -> 水膨胀 -> 密度下降 -> 慢化能力减弱 -> k_eff 下降
        water_density = 0.743 - 0.001 * (self.mod_temp - 600.0)
        water.set_density('g/cm3', water_density)
        
        water.add_s_alpha_beta('c_H_in_H2O')
        water.temperature = self.mod_temp
        
        if self.boron_ppm > 0:
            water.add_element('B', self.boron_ppm * 1e-6)

        self.materials = openmc.Materials([fuel, clad, water])

    def _build_geometry(self):
        fuel_cyl = openmc.ZCylinder(r=self.clad_inner_r)
        clad_cyl = openmc.ZCylinder(r=self.clad_outer_r)
        
        fuel_cell = openmc.Cell(name='Fuel', fill=self.materials[0], region=-fuel_cyl)
        clad_cell = openmc.Cell(name='Clad', fill=self.materials[1], region=+fuel_cyl & -clad_cyl)
        
        pitch = 1.26
        half_pitch = pitch / 2.0
        mod_cell = openmc.Cell(name='Moderator', fill=self.materials[2], region=+clad_cyl)

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
        
        # 开启温度插值，避免因为找不到具体温度截面而报错
        self.settings.temperature = {'method': 'interpolation'}
        
        self.settings.batches = 100
        self.settings.inactive = 20
        self.settings.particles = 10000
        
        source = openmc.IndependentSource()
        source.space = openmc.stats.Point((0, 0, 0))
        self.settings.source = source

    def run_and_extract(self, output_dir: str):
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
        
        model.run(cwd=out_dir_abs)
        
        sp_files = glob.glob(os.path.join(out_dir_abs, 'statepoint.*.h5'))
        if not sp_files:
            raise FileNotFoundError(f"在 {out_dir_abs} 下找不到 statepoint 文件！")
        sp_path = max(sp_files, key=os.path.getctime)
        
        with h5py.File(sp_path, 'r') as f:
            k_combined = f['k_combined'][()]
            k_eff = float(k_combined[0])
            k_std = float(k_combined[1])
            
        return {'k_eff': k_eff, 'std': k_std}