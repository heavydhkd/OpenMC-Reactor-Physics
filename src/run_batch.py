import os
import sys
import time
import yaml
import pandas as pd
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed

# 把项目根目录加入路径，防止 import 报错
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.pin_cell import VeraPinCell

def run_single_case(index, enrichment, boron_ppm, clad_r, fuel_temp, mod_temp):
    out_dir = f"../data/raw/case_{index:03d}"
    try:
        cell = VeraPinCell(
            enrichment=enrichment, 
            boron_ppm=boron_ppm, 
            clad_outer_r=clad_r, 
            fuel_temp=fuel_temp,    
            mod_temp=mod_temp       
        )
        res = cell.run_and_extract(output_dir=out_dir)
        return {
            'case_id': index, 'enrichment': enrichment, 'boron_ppm': boron_ppm, 
            'clad_r': clad_r, 'fuel_temp': fuel_temp, 'mod_temp': mod_temp,
            'k_eff': res['k_eff'], 'std': res['std'], 'status': 'success'
        }
    except Exception as e:
        return {'case_id': index, 'status': f'failed: {str(e)}'}

def main():
    start_time = time.time()
    
    # 1. 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), '../configs/params.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    n_samples = config['sampling']['n_samples']
    seed = config['sampling']['random_state']
    bounds = config['sampling']['bounds']
    
    print(f"====== 开始执行 {n_samples} 组 OpenMC 批量计算 (5 维) ======")
    
    # 2. LHS 采样 (5维)
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # 3. 映射到物理边界
    l_bounds = [
        bounds['enrichment_pct'][0], bounds['boron_ppm'][0], 
        bounds['clad_outer_radius'][0], bounds['fuel_temp'][0], bounds['mod_temp'][0]
    ]
    u_bounds = [
        bounds['enrichment_pct'][1], bounds['boron_ppm'][1], 
        bounds['clad_outer_radius'][1], bounds['fuel_temp'][1], bounds['mod_temp'][1]
    ]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
    
    # 4. 多进程并行执行
    results = []  # 必须在这里初始化结果列表
    max_workers = max(1, os.cpu_count() - 1)
    print(f"启动 {max_workers} 个并行进程...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(n_samples):
            enr, b, clad, f_t, m_t = scaled_samples[i]
            # 传入 5 个参数
            futures.append(executor.submit(run_single_case, i, enr, b, clad, f_t, m_t))
            
        completed = 0
        # 收集结果
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            
            if res['status'] == 'success':
                print(f"[{completed}/{n_samples}] Case {res['case_id']:03d} 完成 | k_eff = {res['k_eff']:.5f} ± {res['std']:.5f}")
            else:
                print(f"[{completed}/{n_samples}] Case {res['case_id']:03d} 失败 | {res['status']}")
    
    # 5. 整理并保存数据
    df = pd.DataFrame(results)
    df = df.sort_values(by='case_id').reset_index(drop=True)
    
    output_csv = os.path.join(os.path.dirname(__file__), '../data/processed/vera_p1_dataset.csv')
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n====== 所有任务结束 ======")
    print(f"总耗时: {elapsed/60:.2f} 分钟")
    print(f"数据集已保存至: {os.path.abspath(output_csv)}")

if __name__ == '__main__':
    main()