import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt

# -------------------------- 数据读取模块 --------------------------
def load_data():
    """读取种植数据、统计数据和参数"""
    # 读取MAT文件数据
    plant2023 = loadmat("2023种植方案.mat")['plant2023']  # 2023年种植方案
    stastic = loadmat("种植相关统计数据.mat")['Stastic']    # 种植统计数据
    output2023 = loadmat("2023种植统计.mat")['output2023']  # 2023年产出数据
    sale = loadmat("各作物各季度售价.mat")['Sale']          # 售价数据
    
    # 读取Excel数据
    data1 = pd.read_excel("1.xlsx")
    area = data1['Sheet1'].values  # 耕地面积数据
    
    return {
        'plant2023': plant2023,
        'stastic': stastic,
        'output2023': output2023,
        'sale': sale,
        'area': area
    }

# -------------------------- 参数定义模块 --------------------------
def init_params():
    """初始化算法参数和问题参数"""
    return {
        # 遗传算法参数
        'n_pop': 50,          # 种群数量
        'max_gen': 500,       # 最大迭代次数
        'crossover_rate': 0.1, # 交叉概率
        'mutation_rate': 0.01, # 变异概率
        'tournament_size': 5,  # 锦标赛选择规模
        
        # 问题参数
        'num_years': 8,       # 年份范围（2023-2030）
        'num_plots': 54,      # 地块数量
        'num_crops': 41,      # 作物总数
        
        # 耕地类型划分（对应原文crop_massif）
        'crop_groups': {
            1: list(range(1, 16)),    # 平旱地/梯田/山坡地作物
            2: [16],                  # 水浇地单季作物
            3: list(range(17, 35)),   # 水浇地第一季等作物
            4: list(range(35, 38)),   # 水浇地第二季作物
            5: list(range(38, 42))    # 普通大棚第二季作物
        },
        
        # 耕地类型矩阵（1-6对应不同类型）
        'land_types': np.concatenate([
            np.ones(6),          # 类型1：6块
            2*np.ones(14),       # 类型2：14块
            3*np.ones(6),        # 类型3：6块
            4*np.ones(8),        # 类型4：8块
            5*np.ones(16),       # 类型5：16块
            6*np.ones(4)         # 类型6：4块
        ]),
        
        # 季节矩阵大小（每行：[季节数, 每季最大作物数]）
        'season_matrix_sizes': {
            1: (1, 1),  # 平旱地：1季1作物
            2: (1, 1),  # 梯田：1季1作物
            3: (1, 1),  # 山坡地：1季1作物
            4: (1, 2),  # 水浇地：最多2作物
            5: (2, 2),  # 普通大棚：2季各2作物
            6: (2, 2)   # 智慧大棚：2季各2作物
        }
    }

# -------------------------- 个体生成与编码 --------------------------
def create_individual(params, data, year):
    """生成单个个体（种植方案）"""
    individual = []
    land_types = params['land_types']
    crop_groups = params['crop_groups']
    season_sizes = params['season_matrix_sizes']
    area = data['area']
    
    for plot in range(params['num_plots']):
        land_type = int(land_types[plot])
        # 确定季节数和每季最大作物数
        num_seasons, max_crops = season_sizes[land_type]
        if land_type == 4:  # 水浇地特殊处理
            num_seasons = random.choice([1, 2])
        
        planting_plan = []
        for season in range(1, num_seasons+1):
            # 确定作物组
            if (land_type == 4 and num_seasons == 1):
                group = 2
            elif (land_type == 5 and num_seasons == 2 and season == 1) or \
                 (land_type in [6, 4] and num_seasons == 2 and season == 1):
                group = 3
            elif land_type == 4 and num_seasons == 2 and season == 2:
                group = 4
            elif land_type == 5 and num_seasons == 2 and season == 2:
                group = 5
            else:
                group = 1
            
            available_crops = crop_groups[group]
            num_crops = random.randint(1, max_crops)  # 随机选择1- max作物
            crops = random.sample(available_crops, num_crops)  # 不重复选择
            planting_plan.append({
                'land_type': land_type,
                'season': season,
                'crops': crops,
                'area': area[plot]  # 地块面积
            })
        individual.append(planting_plan)
    return individual

def init_population(params, data):
    """初始化种群"""
    population = []
    # 初始化2023年数据（基于历史数据）
    initial_plan = data['plant2023']
    for i in range(params['n_pop']):
        # 生成多年种植方案
        individual = []
        for year in range(params['num_years']):
            if year == 0:  # 第一年用历史数据
                individual.append(initial_plan)
            else:  # 后续年份随机生成
                individual.append(create_individual(params, data, year))
        population.append({
            'chromosome': individual,
            'fitness': 0  # 初始适应度
        })
    return population

# -------------------------- 适应度函数（目标函数） --------------------------
def calculate_fitness(individual, params, data):
    """计算个体适应度（利润总和）"""
    total_profit = 0
    sale = data['sale']
    output2023 = data['output2023']
    stastic = data['stastic']
    
    for year_idx in range(1, params['num_years']):  # 从2024年开始计算
        year_plan = individual['chromosome'][year_idx]
        cost = 0
        output = np.zeros((params['num_crops'], 2))  # 作物-季节产出矩阵
        
        for plot in range(params['num_plots']):
            plot_plan = year_plan[plot]
            for season_data in plot_plan:
                season = season_data['season'] - 1  # 转为0索引
                crops = season_data['crops']
                area = season_data['area']
                
                for crop in crops:
                    crop_idx = crop - 1  # 转为0索引
                    # 计算种植成本（元/亩 * 面积）
                    plant_cost = stastic[1, plot, crop_idx, season]
                    cost += area * plant_cost
                    
                    # 计算产量（斤/亩 * 面积）
                    yield_per_mu = stastic[0, plot, crop_idx, season]
                    output[crop_idx, season] += area * yield_per_mu
        
        # 计算销售收入（情况2：超出部分50%降价）
        revenue = 0
        for crop in range(params['num_crops']):
            for season in range(2):
                exp_sale = output2023[crop, season]  # 预期销量
                act_output = output[crop, season]    # 实际产出
                price = sale[crop, season]           # 售价
                
                if act_output <= exp_sale:
                    revenue += act_output * price
                else:
                    # 正常销售部分 + 降价部分
                    revenue += exp_sale * price + (act_output - exp_sale) * price * 0.5
        
        # 年利润 = 收入 - 成本
        year_profit = revenue - cost
        total_profit += year_profit
    
    return total_profit

# -------------------------- 遗传操作模块 --------------------------
def tournament_selection(population, params):
    """锦标赛选择"""
    candidates = random.sample(population, params['tournament_size'])
    # 选择适应度最高的个体
    candidates.sort(key=lambda x: x['fitness'], reverse=True)
    return candidates[0]['chromosome']

def crossover(parent1, parent2, params):
    """交叉操作"""
    child1 = [row.copy() for row in parent1]
    child2 = [row.copy() for row in parent2]
    
    # 随机选择交叉年份
    crossover_year = random.randint(1, params['num_years'] - 1)  # 从第二年开始
    # 随机选择交叉地块
    for plot in range(params['num_plots']):
        if random.random() < params['crossover_rate']:
            # 交换该年份地块的种植方案
            child1[crossover_year][plot], child2[crossover_year][plot] = \
            parent2[crossover_year][plot], parent1[crossover_year][plot]
    return child1, child2

def mutate(individual, params, data):
    """变异操作"""
    mutated = [row.copy() for row in individual]
    if random.random() < params['mutation_rate']:
        # 随机选择变异地块和年份
        plot = random.randint(0, params['num_plots'] - 1)
        year = random.randint(1, params['num_years'] - 1)  # 非初始年
        
        # 重新生成该地块的种植方案
        mutated[year][plot] = create_individual(params, data, year)[plot]
        # 约束修复（简化版）
        mutated = repair_constraints(mutated, params, data, year, plot)
    return mutated

def repair_constraints(individual, params, data, year, plot):
    """修复约束违反（简化版）"""
    plot_plan = individual[year][plot]
    land_type = int(params['land_types'][plot])
    # 检查轮作约束：不能与上一年同作物
    prev_year_plan = individual[year-1][plot]
    prev_crops = [crop for season in prev_year_plan for crop in season['crops']]
    
    for season_data in plot_plan:
        current_crops = season_data['crops']
        # 移除与上一年重复的作物
        new_crops = [c for c in current_crops if c not in prev_crops]
        # 若作物为空，随机补充
        if not new_crops:
            group = get_crop_group(land_type, season_data['season'], params)
            new_crops = random.sample(params['crop_groups'][group], 1)
        season_data['crops'] = new_crops
    return individual

def get_crop_group(land_type, season, params):
    """获取作物组（辅助函数）"""
    if land_type == 4 and season == 1:
        return 2
    elif (land_type == 5 and season == 1) or (land_type in [6, 4] and season == 1):
        return 3
    elif land_type == 4 and season == 2:
        return 4
    elif land_type == 5 and season == 2:
        return 5
    else:
        return 1

# -------------------------- 主算法流程 --------------------------
def genetic_algorithm(params, data):
    """遗传算法主流程"""
    population = init_population(params, data)
    best_fitness_history = []
    
    for gen in range(params['max_gen']):
        # 计算适应度
        for individual in population:
            individual['fitness'] = calculate_fitness(individual, params, data)
        
        # 记录最优适应度
        best_individual = max(population, key=lambda x: x['fitness'])
        best_fitness_history.append(best_individual['fitness'])
        print(f"迭代次数 {gen+1}/{params['max_gen']}, 最优利润: {best_individual['fitness']:.2f} 元")
        
        # 生成子代
        offspring = []
        while len(offspring) < params['n_pop']:
            # 选择双亲
            parent1 = tournament_selection(population, params)
            parent2 = tournament_selection(population, params)
            
            # 交叉
            if random.random() < params['crossover_rate']:
                child1_chrom, child2_chrom = crossover(parent1, parent2, params)
            else:
                child1_chrom, child2_chrom = parent1.copy(), parent2.copy()
            
            # 变异
            child1_chrom = mutate(child1_chrom, params, data)
            child2_chrom = mutate(child2_chrom, params, data)
            
            # 添加到子代
            offspring.append({'chromosome': child1_chrom, 'fitness': 0})
            offspring.append({'chromosome': child2_chrom, 'fitness': 0})
        
        # 合并种群并筛选
        population += offspring
        # 按适应度排序并保留最优个体
        population.sort(key=lambda x: x['fitness'], reverse=True)
        population = population[:params['n_pop']]
    
    # 绘制进化曲线
    plt.plot(best_fitness_history)
    plt.xlabel('迭代次数')
    plt.ylabel('最优利润 (元)')
    plt.title('适应度进化曲线')
    plt.show()
    
    return best_individual

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 加载数据和参数
    data = load_data()
    params = init_params()
    
    # 运行遗传算法
    best_solution = genetic_algorithm(params, data)
    print(f"最优种植方案总利润: {best_solution['fitness']:.2f} 元")