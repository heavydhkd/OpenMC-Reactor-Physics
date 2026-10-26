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

def run_single_case(index, enrichment, boron_ppm, clad_r):
    """单个算例的运行逻辑"""
    # 将每次运行的输出放到单独的文件夹中，避免互相覆盖
    out_dir = f"../data/raw/case_{index:03d}"
    
    try:
        # 实例化模型 (温度固定为 900K 燃料, 600K 慢化剂)
        cell = VeraPinCell(enrichment=enrichment, boron_ppm=boron_ppm, 
                           clad_outer_r=clad_r, fuel_temp=900.0, mod_temp=600.0)
        
        # 运行并提取数据
        res = cell.run_and_extract(output_dir=out_dir)
        
        return {
            'case_id': index,
            'enrichment': enrichment,
            'boron_ppm': boron_ppm,
            'clad_r': clad_r,
            'k_eff': res['k_eff'],
            'std': res['std'],
            'status': 'success'
        }
    except Exception as e:
        # 如果某一个算例跑飞了，记录下来，不影响其他算例
        return {
            'case_id': index,
            'enrichment': enrichment,
            'boron_ppm': boron_ppm,
            'clad_r': clad_r,
            'status': f'failed: {str(e)}'
        }

def main():
    start_time = time.time()
    
    # 1. 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), '../configs/params.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    n_samples = config['sampling']['n_samples']
    seed = config['sampling']['random_state']
    bounds = config['sampling']['bounds']
    
    print(f"====== 开始执行 {n_samples} 组 OpenMC 批量计算 ======")
    
    # 2. LHS 采样 (3维)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # 3. 映射到物理边界
    l_bounds = [bounds['enrichment_pct'][0], bounds['boron_ppm'][0], bounds['clad_outer_radius'][0]]
    u_bounds = [bounds['enrichment_pct'][1], bounds['boron_ppm'][1], bounds['clad_outer_radius'][1]]
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)
    
    # 4. 并行执行
    results = []
    # max_workers 建议设为 CPU 核心数减 1，防止把系统卡死
    max_workers = max(1, os.cpu_count() - 1)
    print(f"检测到 {os.cpu_count()} 个 CPU 核心，启动 {max_workers} 个并行进程...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(n_samples):
            enr = scaled_samples[i, 0]
            b = scaled_samples[i, 1]
            clad = scaled_samples[i, 2]
            futures.append(executor.submit(run_single_case, i, enr, b, clad))
            
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
    # 按 case_id 排序，方便查看
    df = df.sort_values(by='case_id').reset_index(drop=True)
    
    output_csv = os.path.join(os.path.dirname(__file__), '../data/processed/vera_p1_dataset.csv')
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n====== 所有任务结束 ======")
    print(f"总耗时: {elapsed/60:.2f} 分钟")
    print(f"数据集已保存至: {os.path.abspath(output_csv)}")

if __name__ == '__main__':
    main()