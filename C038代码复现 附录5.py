import numpy as np
import random
import pandas as pd

# 读取数据（请根据实际文件路径调整）
land_data = pd.read_excel("D:!桌面附件1(1).xlsx")  # 地块信息（包含地块名称、地块面积、地块类型等）
crop_data = pd.read_excel(r"D:桌面附件2(1).xlsx")  # 作物数据（包含作物编号、销售单价、种植成本、亩产量等）
demand_data = pd.read_excel(r"D:桌面农作物每季需求量带作物编号(1).xlsx")  # 需求量数据（包含作物编号及对应需求量）

# 数据预处理：提取土地类型和作物类型
## 去除地块名称中的空格并去重
land_types = land_data["地块名称"].str.strip().unique().tolist()
## 生成作物组合编号（作物编号+地块类型）并去重
crop_data["组合编号"] = crop_data["作物编号"].astype(str) + "_" + crop_data["地块类型"].astype(str)
crop_types = crop_data["组合编号"].unique().tolist()

# 定义时间范围（2024-2030年）和季节（1-2季）
years = range(2024, 2031)  # 7年规划期
seasons = range(1, 3)  # 每年2个季节

# 算法核心参数设置
POPULATION_SIZE = 100  # 种群规模
NUM_GENERATIONS = 500  # 迭代次数
MUTATION_FACTOR = 0.8  # 差分进化变异因子
CROSSOVER_RATE = 0.8  # 交叉概率
EARLY_STOPPING_THRESHOLD = 0.01  # 早停阈值（当连续两代最优适应度变化小于该值时停止迭代）

# 种群初始化函数
def initialize_population():
    """初始化种群，每个个体包含种植面积(x)和作物选择(z)两个决策变量"""
    population = []
    for _ in range(POPULATION_SIZE):
        # 初始化决策变量：x为种植面积（维度：土地类型×作物类型×季节×年份）
        # z为二元选择变量（1表示种植，0表示不种植，维度与x一致）
        individual = {
            'x': np.zeros((len(land_types), len(crop_types), len(seasons), len(years))),
            'z': np.random.randint(0, 2, (len(land_types), len(crop_types), len(seasons), len(years)))
        }
        # 初始化种植面积：随机选择地块面积的50%或100%（确保满足最小种植面积约束）
        for i, land in enumerate(land_types):
            land_area = land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0]  # 获取地块总面积
            for k in range(len(crop_types)):
                for j in range(len(seasons)):
                    for t in range(len(years)):
                        individual['x'][i, k, j, t] = random.choice([0.5, 1.0]) * land_area
        population.append(individual)
    return population

# 差分进化变异操作函数
def differential_mutation(population):
    """基于差分进化算法生成变异个体"""
    new_population = []
    for i, target in enumerate(population):
        # 随机选择3个不同的个体用于变异
        indices = list(range(len(population)))
        indices.remove(i)  # 排除当前目标个体
        r1, r2, r3 = random.sample(indices, 3)
        # 生成供体个体（变异向量）
        donor = {
            'x': np.copy(population[r1]['x']),
            'z': np.copy(population[r1]['z'])
        }
        # 差分变异公式：donor = r1 + F*(r2 - r3)
        donor['x'] = population[r1]['x'] + MUTATION_FACTOR * (population[r2]['x'] - population[r3]['x'])
        # 约束种植面积在合理范围（50%-100%地块面积）
        for i in range(len(land_types)):
            land_area = land_data.loc[land_data["地块名称"] == land_types[i], "地块面积"].values[0]
            donor['x'][i] = np.clip(donor['x'][i], 0.5 * land_area, 1.0 * land_area)
        new_population.append(donor)
    return new_population

# 交叉操作函数
def crossover(target, donor):
    """对目标个体和供体个体进行交叉，生成试验个体"""
    trial = {'x': np.copy(target['x']), 'z': np.copy(target['z'])}  # 初始化试验个体为目标个体
    # 按概率进行交叉替换
    for i in range(len(land_types)):
        if random.random() > CROSSOVER_RATE:  # 交叉概率外的位置替换为供体基因
            trial['x'][i] = donor['x'][i]
    return trial

# 选择操作函数
def selection(target, trial, fitness_func):
    """选择适应度更高的个体保留到下一代"""
    target_fitness = fitness_func(target)
    trial_fitness = fitness_func(trial)
    return trial if trial_fitness > target_fitness else target



"""以上是补全的主体"""
# 创建作物相关性矩阵
def create_correlation_matrix():
    # 这里以作物数量为len(crop_types)的方阵为例
    correlation_matrix = np.identity(len(crop_types))  # 初始化为单位矩阵，表示自相关性为1
    # 手动设置作物之间的相关性（可替代性/互补性）
    for i in range(len(crop_types)):
        for j in range(i + 1, len(crop_types)):
            # 根据农作物的替代性/互补性设置相关性
            if "蔬菜" in crop_types[i] and "蔬菜" in crop_types[j]:
                correlation_matrix[i, j] = 0.7  # 蔬菜类作物较高的可替代性
                correlation_matrix[j, i] = 0.7  # 对称矩阵
            elif "粮食" in crop_types[i] and "粮食" in crop_types[j]:
                correlation_matrix[i, j] = 0.5  # 粮食类作物中等替代性
                correlation_matrix[j, i] = 0.5
            else:
                correlation_matrix[i, j] = 0.2  # 其他作物的低相关性
                correlation_matrix[j, i] = 0.2
    return correlation_matrix

# 适应度函数：加入CVaR和相关性分析
def calculate_fitness_with_cvar_and_correlation(individual, correlation_matrix, alpha=0.95):
    fitness = 0
    penalty = 0
    losses = []
    for i, land in enumerate(land_types):
        for k, crop in enumerate(crop_types):
            for j in seasons:
                for t in years:
                    area = individual['x'][i, k, j - 1, t - 2024]
                    z_val = individual['z'][i, k, j - 1, t - 2024]
                    crop_info = crop_data[crop_data["组合编号"] == crop]
                    if not crop_info.empty:
                        base_price = crop_info["销售单价"].values[0]
                        base_cost = crop_info["种植成本"].values[0]
                        base_yield_per_acre = crop_info["亩产量/斤"].values[0]

                        # 相关性对价格和成本的影响
                        price_correlation_factor = 1 + correlation_matrix[k].mean()  # 相关性对价格的影响
                        cost_correlation_factor = 1 + correlation_matrix[k].mean()  # 相关性对种植成本的影响

                        # 不同农作物的价格趋势
                        if "粮食" in crop:  # 粮食类作物价格稳定
                            price = base_price * price_correlation_factor
                        elif "蔬菜" in crop:  # 蔬菜类作物每年价格上涨5%
                            price = base_price * (1.05 ** (t - 2023)) * price_correlation_factor
                        elif "食用菌" in crop:  # 食用菌价格每年下降1%-5%
                            price_decline = random.uniform(0.01, 0.05)
                            price = base_price * (1 - price_decline) ** (t - 2023) * price_correlation_factor
                        else:
                            price = base_price * price_correlation_factor

                        # 考虑种植成本的年增长率为5%并且受相关性影响
                        cost = base_cost * (1.05 ** (t - 2023)) * cost_correlation_factor

                        # 考虑产量的波动范围为±10%并且受相关性影响
                        yield_per_acre = base_yield_per_acre * (1 + random.uniform(-0.1, 0.1))

                        # 需求量的变化范围为±5%
                        demand = demand_data[demand_data["作物编号"] == int(crop.split()[0])]["需求量"].values[0]
                        demand = demand * (1 + random.uniform(-0.05, 0.05))

                        # 计算收益
                        profit = min(yield_per_acre * area, demand) * price - cost * area
                        fitness += profit

                        # 记录损失（负收益）
                        if profit < 0:
                            losses.append(abs(profit))

                        # 约束条件1：地块总面积限制
                        if np.sum(individual['x'][i, :, j - 1, t - 2024]) > land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0]:
                            penalty += 1000  # 违反面积限制的罚则

                        # 约束条件2：单块地种植面积下限
                        if area <= 0.5 * land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0] and z_val == 1:
                            penalty += 500  # 种植面积不足的罚则

                        # 约束条件3：作物种植与二元变量关联
                        if area > land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0] * z_val:
                            penalty += 1000  # 违反种植与二元变量关联

    # CVaR惩罚项计算
    if losses:
        sorted_losses = sorted(losses)
        index = int(np.ceil(alpha * len(sorted_losses)))
        cvar = np.mean(sorted_losses[:index])  # 计算CVaR
        penalty += cvar  # 将CVaR作为惩罚项

    return fitness - penalty

# 主算法流程
def differential_evolution():
    population = initialize_population()
    best_fitness_history = []
    correlation_matrix = create_correlation_matrix()  # 创建相关性矩阵
    for generation in range(NUM_GENERATIONS):
        new_population = []
        for i in range(POPULATION_SIZE):
            donor = differential_mutation(population)[i]  # 生成变异个体
            trial = crossover(population[i], donor)  # 交叉生成新的个体
            selected_individual = selection(population[i], trial)  # 选择最优
            new_population.append(selected_individual)
        # 替换种群
        population = new_population
        # 计算每一代的最优适应度
        fitness_values = [calculate_fitness_with_cvar_and_correlation(ind, correlation_matrix) for ind in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        # 早期停止机制
        if generation > 0 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping at generation {generation}")
            break
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    return max(population, key=lambda ind: calculate_fitness_with_cvar_and_correlation(ind, correlation_matrix))