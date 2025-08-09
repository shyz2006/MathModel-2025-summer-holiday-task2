import numpy as np
import random
import pandas as pd

# 读取数据
land_data = pd.read_excel("D:!桌面附件1(1).xlsx")  # 地块信息
crop_data = pd.read_excel(r"D:桌面附件2(1).xlsx")  # 作物产量/成本/价格
demand_data = pd.read_excel(r"D:桌面农作物每季需求量带作物编号(1).xlsx")  # 需求量

# 去除空格并确保唯一
land_types = land_data["地块名称"].str.strip().unique().tolist()
crop_data["组合编号"] = crop_data["作物编号"].astype(str) + crop_data["地块类型"]
crop_types = crop_data["组合编号"].unique().tolist()

years = range(2024, 2031)
seasons = range(1, 3)

# 参数设置
POPULATION_SIZE = 100  # 种群规模
NUM_GENERATIONS = 500  # 迭代次数
MUTATION_FACTOR = 0.8  # 变异因子
CROSSOVER_RATE = 0.8  # 交叉率
EARLY_STOPPING_THRESHOLD = 0.01  # 停止阈值

# 初始化种群
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = {
            'x': np.zeros((len(land_types), len(crop_types), len(seasons), len(years))),
            'z': np.random.randint(0, 2, (len(land_types), len(crop_types), len(seasons), len(years)))
        }
        # 初始化解，面积只能是50%或100%
        for i, land in enumerate(land_types):
            land_area = land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0]
            for k in range(len(crop_types)):
                for j in range(len(seasons)):
                    for t in range(len(years)):
                        # 随机选择50%或100%的面积
                        individual['x'][i, k, j, t] = random.choice([0.5, 1.0]) * land_area
        population.append(individual)
    return population

# 适应度函数：基于目标函数计算每个个体的适应度
def calculate_fitness(individual):
    fitness = 0
    penalty = 0
    for i, land in enumerate(land_types):
        for k, crop in enumerate(crop_types):
            for j in seasons:
                for t in years:
                    area = individual['x'][i, k, j - 1, t - 2024]
                    z_val = individual['z'][i, k, j - 1, t - 2024]
                    crop_info = crop_data[crop_data["组合编号"] == crop]
                    if not crop_info.empty:
                        price = crop_info["销售单价"].values[0]
                        cost = crop_info["种植成本"].values[0]
                        yield_per_acre = crop_info["亩产量/斤"].values[0]
                        demand = demand_data[demand_data["作物编号"] == int(crop.split()[0])]["需求量"].values[0]
                        profit = min(yield_per_acre * area, demand) * price - cost * area
                        fitness += profit

                        # 约束条件1：地块总面积限制
                        if np.sum(individual['x'][i, :, j - 1, t - 2024]) > land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0]:
                            penalty += 1000  # 违反面积限制的罚则

                        # 约束条件2：单块地种植面积下限
                        if area <= 0.5 * land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0] and z_val == 1:
                            penalty += 500  # 种植面积不足的罚则

                        # 约束条件3：作物种植与二元变量关联
                        if area > land_data.loc[land_data["地块名称"] == land, "地块面积"].values[0] * z_val:
                            penalty += 1000  # 违反种植与二元变量关联
    return fitness - penalty

# 差分进化变异操作
def differential_mutation(population):
    new_population = []
    for i, target in enumerate(population):
        indices = list(range(len(population)))
        indices.remove(i)
        r1, r2, r3 = random.sample(indices, 3)
        donor = {
            'x': np.copy(population[r1]['x']),
            'z': np.copy(population[r1]['z'])
        }
        donor['x'] = population[r1]['x'] + MUTATION_FACTOR * (population[r2]['x'] - population[r3]['x'])
        # 确保生成的解在合理范围内，面积只能是50%或100%
        for i in range(len(land_types)):
            land_area = land_data.loc[land_data["地块名称"] == land_types[i], "地块面积"].values[0]
            donor['x'][i] = np.clip(donor['x'][i], 0.5 * land_area, 1.0 * land_area)
        new_population.append(donor)
    return new_population

# 交叉操作
def crossover(target, donor):
    trial = {'x': np.copy(target['x']), 'z': np.copy(target['z'])}
    for i in range(len(land_types)):
        if random.random() > CROSSOVER_RATE:
            trial['x'][i] = donor['x'][i]  # 交叉替换
    return trial

# 选择操作：选择适应度较好的个体
def selection(target, trial):
    target_fitness = calculate_fitness(target)
    trial_fitness = calculate_fitness(trial)
    return trial if trial_fitness > target_fitness else target

# 主算法流程
def differential_evolution():
    population = initialize_population()
    best_fitness_history = []
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
        fitness_values = [calculate_fitness(ind) for ind in population]
        best_fitness = max(fitness_values)
        best_fitness_history.append(best_fitness)
        # 早期停止机制
        if generation > 0 and abs(best_fitness_history[-1] - best_fitness_history[-2]) < EARLY_STOPPING_THRESHOLD:
            print(f"Early stopping at generation {generation}")
            break
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    return max(population, key=calculate_fitness)

# 运行算法
best_solution = differential_evolution()

# 输出最优结果
output_data = []
for i, land in enumerate(land_types):
    for k, crop in enumerate(crop_types):
        for j in seasons:
            for t in years:
                area = best_solution['x'][i, k, j - 1, t - 2024]
                if area > 0:
                    output_data.append((land, crop, j, t, area))

# 保存结果到Excel
print(output_data)