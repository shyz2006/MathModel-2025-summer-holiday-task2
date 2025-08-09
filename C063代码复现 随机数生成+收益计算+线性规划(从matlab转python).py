import numpy as np
import pandas as pd
"""随机数生成"""
# 生成1000组状态对应的参数
K = 1000  # 随机状态数量
# 初始化随机数矩阵（4列分别对应：小麦/玉米销量变化率、其他作物销量变化率、亩产量变化率、食用菌价格变化率）
A = np.zeros((K, 4))  

for i in range(K):
    # 小麦/玉米预期销售量变化率（5%-10%）
    salesvolume1 = np.random.uniform(5, 10)  
    # 其他作物销售量变化率（-5%~5%）
    salesvolume2 = np.random.uniform(-5, 5)  
    # 亩产量变化率（-10%~10%）
    outp = np.random.uniform(-10, 10)  
    # 食用菌价格变化率（-5%~-1%）
    salesprice = np.random.uniform(-5, -1)  
    # 转换为小数（除以100）
    A[i, :] = [salesvolume1, salesvolume2, outp, salesprice] / 100  

# 保存到Excel（路径需根据实际情况修改）
df = pd.DataFrame(A, columns=['小麦玉米销量变化率', '其他作物销量变化率', '亩产量变化率', '食用菌价格变化率'])
df.to_excel('随机数生成.xlsx', index=False)
print("随机数生成完成，已保存至Excel")



"""收益计算"""
import numpy as np
import pandas as pd

# 读取种植方案数据（7年数据，路径需修改）
def read_planting_plan():
    fang_an = []
    for year in range(1, 8):  # 2024-2030年
        # 读取每个年份的Excel表（sheet1至sheet7）
        df = pd.read_excel(
            '种植方案数据.xlsx', 
            sheet_name=f'sheet{year}', 
            usecols='C:AQ',  # 对应原代码中的C2:AQ109
            header=None
        )
        fang_an.append(df.values)  # 转换为numpy数组存储
    return fang_an

# 读取利润相关参数（路径需修改）
def read_profit_params():
    cost = pd.read_excel('参数表.xlsx', sheet_name='Sheet3', usecols='C', header=None).values.flatten()  # 种植成本
    exp_sale1 = pd.read_excel('参数表.xlsx', sheet_name='Sheet4', usecols='C', header=None).values.flatten()  # 第一季预期销量
    exp_sale2 = pd.read_excel('参数表.xlsx', sheet_name='Sheet4', usecols='D', header=None).values.flatten()  # 第二季预期销量
    per_out = pd.read_excel('参数表.xlsx', sheet_name='Sheet2', usecols='C:AQ', header=None).values  # 亩产量
    price1 = pd.read_excel('参数表.xlsx', sheet_name='Sheet5', usecols='E', header=None).values.flatten()  # 第一季价格
    price2 = pd.read_excel('参数表.xlsx', sheet_name='Sheet5', usecols='F', header=None).values.flatten()  # 第二季价格
    return cost, exp_sale1, exp_sale2, per_out, price1, price2

# 计算各状态下的收益
def calculate_profit():
    fang_an = read_planting_plan()
    cost, exp_sale1, exp_sale2, per_out, price1, price2 = read_profit_params()
    random_g = pd.read_excel('随机数生成.xlsx').values  # 读取随机数
    H = random_g.shape[0]  # 随机状态数量
    pro_mat = np.zeros((H, 8))  # 存储每年利润和总利润

    for i in range(H):
        rg = random_g[i, :]  # 第i组随机数
        for j in range(7):  # 7年
            # 更新预期销售量（小麦/玉米年增长，其他作物波动）
            new_exp_sale1 = exp_sale1 * (1 + rg[1])
            new_exp_sale1[[5, 6]] = exp_sale1[[5, 6]] * (1 + rg[0]) ** (j + 1)  # 小麦/玉米（索引5、6对应原代码j=6,7）
            new_exp_sale2 = exp_sale2 * (1 + rg[1])
            new_exp_sale2[[5, 6]] = exp_sale2[[5, 6]] * (1 + rg[0]) ** (j + 1)

            # 更新亩产量和成本
            new_per_out = per_out * (1 + rg[2])  # 亩产量变化
            new_cost = cost * (1 + 0.05) ** (j + 1)  # 种植成本年增5%

            # 更新销售价格（蔬菜年增5%，食用菌波动，羊肚菌年降5%）
            new_price1 = price1.copy()
            new_price1[16:37] *= (1 + 0.05) ** (j + 1)  # 蔬菜类（索引16-36对应原j=17-37）
            new_price1[37:40] *= (1 + rg[3])  # 食用菌（索引37-39对应原j=38-40）
            new_price1[40] *= (1 - 0.05) ** (j + 1)  # 羊肚菌（索引40对应原j=41）

            new_price2 = price2.copy()
            new_price2[16:37] *= (1 + 0.05) ** (j + 1)
            new_price2[37:40] *= (1 + rg[3])
            new_price2[40] *= (1 - 0.05) ** (j + 1)

            # 计算产量和销售额
            output = fang_an[j] * new_per_out  # 地块产量
            crops_output = np.vstack([
                np.sum(output[:54, :], axis=0),  # 第一季总产量
                np.sum(output[54:108, :], axis=0)  # 第二季总产量
            ])
            new_exp_sale = np.vstack([new_exp_sale1, new_exp_sale2])  # 总预期销量
            sales_q = np.minimum(crops_output, new_exp_sale)  # 实际销量（取产量与预期最小值）
            # 销售额（含滞销部分半价销售）
            sales_v = (sales_q + 0.5 * np.maximum(0, crops_output - new_exp_sale)) * np.vstack([new_price1, new_price2])

            # 计算成本和利润
            cost_u = fang_an[j] * new_cost  # 地块成本
            cost_j = np.vstack([
                np.sum(cost_u[:54, :], axis=0),  # 第一季总成本
                np.sum(cost_u[54:108, :], axis=0)  # 第二季总成本
            ])
            profit = np.sum(sales_v - cost_j)  # 当年利润
            pro_mat[i, j] = profit

        pro_mat[i, 7] = np.sum(pro_mat[i, :7])  # 7年总利润

    # 保存结果
    pd.DataFrame(pro_mat, columns=[f'第{y+1}年利润' for y in range(7)] + ['7年总利润']).to_excel('收益计算结果.xlsx', index=False)
    print("收益计算完成，已保存至Excel")

if __name__ == "__main__":
    calculate_profit()



"""线性规划"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# 读取地块和作物参数
def read_lp_params():
    # 地块面积（54个地块）
    a = pd.read_excel('参数表.xlsx', sheet_name='sheet1', usecols='C', header=None).values.flatten()[:54]
    # 作物成本和价格（简化为均值）
    cost = pd.read_excel('参数表.xlsx', sheet_name='Sheet3', usecols='C', header=None).values.flatten()[:41]  # 41种作物
    price = pd.read_excel('参数表.xlsx', sheet_name='Sheet5', usecols='E', header=None).values.flatten()[:41]
    return a, cost, price

# 构建线性规划模型
def lp_model():
    a, cost, price = read_lp_params()
    n_lots = len(a)  # 地块数量：54
    n_crops = len(cost)  # 作物数量：41
    n_years = 7  # 年份：2024-2030

    # 变量维度：地块×作物×年份×2（第一季X和第二季Y），此处简化为单一年份的X变量
    # 目标函数：最大化利润（收入-成本）
    # 收入 = 价格×销量，成本 = 单位成本×种植面积
    # 简化目标：max (price[j] * X[i,j] - cost[j] * X[i,j]) 对所有i,j
    c = -(price - cost)  # 线性规划默认求最小值，故取负号

    # 约束条件1：地块面积约束（X[i,j]总和 ≤ 地块面积a[i]）
    A_eq = []
    b_eq = []
    for i in range(n_lots):
        row = np.zeros(n_crops)
        row[i * n_crops : (i+1) * n_crops] = 1  # 每个地块的作物面积总和
        A_eq.append(row)
        b_eq.append(a[i])

    # 其他约束（轮作、作物类型匹配等需进一步细化，此处简化）
    # 求解
    res = linprog(c, A_ub=A_eq, b_ub=b_eq, method='highs')
    if res.success:
        print(f"最优总利润：{-res.fun:.2f}元")
        # 保存最优种植面积
        pd.DataFrame(res.x.reshape(n_lots, n_crops)).to_excel('最优种植方案.xlsx', index=False)
    else:
        print("求解失败")

if __name__ == "__main__":
    lp_model()