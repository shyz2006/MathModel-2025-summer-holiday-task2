# coding:utf-8
import pandas as pd
import numpy as np

# 读取Excel文件中的特定工作表
filepath1 = r"E:\存放桌面\数学建模资料汇总数学建模BOOM\2025暑期任务-2\C题\附件1.xlsx"
filepath2 = r"E:\存放桌面\数学建模资料汇总数学建模BOOM\2025暑期任务-2\C题\附件2.xlsx"
data1_land_info = pd.read_excel(filepath1, sheet_name="乡村的现有耕地")
data1_zw_info = pd.read_excel(filepath1, sheet_name="乡村种植的农作物")
data2_land_2023_info = pd.read_excel(filepath2, sheet_name="2023年的农作物种植情况")
data2_zw_2023_info = pd.read_excel(filepath2, sheet_name="2023年统计的相关数据")

land_info = data1_land_info[['地块名称', '地块类型', '地块面积/亩']]
zw_info = data1_zw_info[["作物编号", "作物名称", "作物类型"]]
land_2023_info = data2_land_2023_info[["种植地块", "作物编号", "作物名称", "作物类型", "种植面积/亩", "种植季次"]]
zw_2023_info = data2_zw_2023_info[['作物编号', '作物名称', '地块类型', '种植季次', '亩产量/斤', '种植成本/元/亩', '销售单价/元/斤', '单价']]
zw_2023_info["性价比"] = (zw_2023_info['亩产量/斤'] * zw_2023_info['销售单价/元/斤']) - zw_2023_info['种植成本/元/亩']

planting_history = {}

def initialize_planting_history(land_2023_info):
    for index, row in land_2023_info.iterrows():
        land_name = row['种植地块']
        season = row['种植季次']
        crop_name = row['作物名称']
        area = row['种植面积/亩']
        if land_name not in planting_history:
            planting_history[land_name] = {}
        if 2023 not in planting_history[land_name]:
            planting_history[land_name][2023] = {}
        planting_history[land_name][2023][season] = {"作物名称": crop_name, "种植面积/亩": area}

# 初始化2023年数据
initialize_planting_history(land_2023_info)

# 记录每年的种植信息
def record_planting(year, land_name, season, crop_name, allocated_area):
    if land_name not in planting_history:
        planting_history[land_name] = {}
    if year not in planting_history[land_name]:
        planting_history[land_name][year] = {}
    planting_history[land_name][year][season] = {"作物名称": crop_name, "种植面积/亩": allocated_area}

# 定义函数计算每个地块的产量
def find_yield(row):
    match = zw_2023_info[
        (zw_2023_info['作物编号'] == row['作物编号']) &
        (zw_2023_info['地块类型'] == row['地块类型']) &
        (zw_2023_info['种植季次'] == row['种植季次'])
    ]
    if not match.empty:
        return match.loc[0]['亩产量/斤'] * row['种植面积/亩']
    else:
        return None

# 计算总产量
data2_land_2023_info['总产量/斤'] = data2_land_2023_info.apply(find_yield, axis=1)
crop_total_yield = data2_land_2023_info.groupby(["作物编号", "作物名称", "种植季次"])['总产量/斤'].sum().reset_index()

# 根据地块类型、种植季次、作物编号等进行分组，并按照性价比降序排序
land_crop_efficiency_rank = zw_2023_info.groupby(["地块类型", "种植季次"]).apply(
    lambda x: x.sort_values("性价比", ascending=False)
).reset_index(drop=True)

output_file = "性价比排行榜.xlsx"  # 将性价比结果保存到Excel文件中
land_crop_efficiency_rank.to_excel(output_file, index=False)
print(f"性价比结果已保存至{output_file}")

def greedy_crop_strategy_ABC_DEF(land_info, crop_efficiency_rank, crop_total_yield_initial):
    # 贪心算法实现: 生成2024-2030年的种植策略
    total_profit = 0
    years = range(2024, 2031)
    planting_plan = []
    last_year_crop = {land_name: None for land_name in land_info['地块名称']}

    for year in years:
        year_profit = 0  # 每年的收益
        # 初始化地块的剩余面积
        land_info["剩余面积/亩"] = land_info['地块面积/亩']
        crop_total_yield = crop_total_yield_initial.copy()

        # 遍历每块土地
        for idx, land in land_info.iterrows():
            land_name = land['地块名称']
            land_type = land['地块类型']
            remaining_area = land["剩余面积/亩"]

            # 对于ABC类土地的处理逻辑
            if land_name[0] in ['A', 'B', 'C']:
                beans = ["黄豆", "黑豆", "红豆", "绿豆", "爬豆"]
                # 如果是2025年及以后，检查前两年豆类种植面积
                if year >= 2025:
                    past_two_years_bean_area = 0
                    # 计算前两年的豆类作物种植面积
                    for past_year in range(year - 2, year):
                        if land_name in planting_history and past_year in planting_history[land_name]:
                            for season in planting_history[land_name][past_year].values():
                                if season['作物名称'] in beans:
                                    past_two_years_bean_area += season['种植面积/亩']
                    # 如果前两年豆类种植面积小于当前地块面积，则优先补种豆类
                    if past_two_years_bean_area < land['地块面积/亩']:
                        required_bean_area = land["地块面积/亩"] - past_two_years_bean_area
                        # 获取性价比最高且需求未饱和的豆类作物
                        available_beans = crop_efficiency_rank[
                            (crop_efficiency_rank["作物名称"].isin(beans)) &
                            (crop_efficiency_rank["地块类型"] == land_type)
                        ]
                        for _, bean_crop in available_beans.iterrows():
                            bean_crop_name = bean_crop['作物名称']
                            bean_crop_id = bean_crop['作物编号']
                            bean_efficiency = bean_crop['性价比']
                            # 获取豆类作物的需求
                            bean_demand = crop_total_yield[
                                (crop_total_yield["作物编号"] == bean_crop_id)
                            ]
                            if not bean_demand.empty:
                                total_demand = bean_demand["总产量/斤"].values[0]
                                if total_demand > 0 and required_bean_area > 0:
                                    # 分配豆类作物的种植面积
                                    allocated_area = min(required_bean_area, total_demand / bean_crop['亩产量/斤'])
                                    total_demand -= allocated_area * bean_crop['亩产量/斤']
                                    remaining_area -= allocated_area
                                    required_bean_area -= allocated_area
                                    year_profit += allocated_area * bean_efficiency  # 计算收益

                                    # 更新豆类需求
                                    crop_total_yield.loc[
                                        crop_total_yield["作物编号"] == bean_crop_id, "总产量/斤"
                                    ] = total_demand

                                    # 保存豆类种植记录
                                    planting_plan.append({
                                        '地块名称': land_name,
                                        '作物编号': bean_crop_id,
                                        '作物名称': bean_crop_name,
                                        '种植面积/亩': allocated_area,
                                        '种植季次': f"{year}第1季",
                                        '性价比': bean_efficiency
                                    })
                                    record_planting(year, land_name, f"第1季", bean_crop_name, allocated_area)  # 记录种植信息
                                    if remaining_area == 0:
                                        break  # 地块种植面积已满，退出
                        if remaining_area == 0:
                            continue

                # 如果豆类补种后仍有剩余面积，则继续种植性价比高的其他作物
                if remaining_area > 0:
                    available_crops = crop_efficiency_rank[
                        (crop_efficiency_rank["地块类型"] == land_type)
                    ]
                    # 检查前一年种植的作物，确保不重复种植
                    last_year_crop_name = None
                    if year > 2023:
                        last_year_crop_name = planting_history.get(land_name, {}).get(year - 1, {}).get(
                            "第1季", {}).get('作物名称', None)
                    for _, crop in available_crops.iterrows():
                        crop_name = crop['作物名称']
                        crop_id = crop['作物编号']
                        crop_efficiency = crop['性价比']
                        # 检查是否与前一年种植的作物相同
                        if crop_name == last_year_crop_name:
                            continue  # 如果相同，跳过该作物
                        # 获取该作物的需求
                        crop_demand = crop_total_yield[
                            (crop_total_yield["作物编号"] == crop_id)
                        ]
                        if not crop_demand.empty:
                            total_demand = crop_demand['总产量/斤'].values[0]
                            if total_demand > 0:
                                while remaining_area > 0 and total_demand > 0:
                                    # 分配给作物的面积
                                    allocated_area = min(remaining_area, total_demand / crop['亩产量/斤'])
                                    total_demand -= allocated_area * crop['亩产量/斤']
                                    remaining_area -= allocated_area
                                    # 计算收益
                                    year_profit += allocated_area * crop_efficiency
                                    # 更新需求
                                    crop_total_yield.loc[
                                        (crop_total_yield["作物编号"] == crop_id), "总产量/斤"
                                    ] = total_demand
                                    # 保存分配记录
                                    planting_plan.append({
                                        "地块名称": land_name,
                                        "作物编号": crop_id,
                                        "作物名称": crop_name,
                                        "种植面积/亩": allocated_area,
                                        "种植季次": f"{year}第1季",
                                        "性价比": crop_efficiency
                                    })
                                    # 记录当前种植信息
                                    record_planting(year, land_name, "第1季", crop_name, allocated_area)
                                    if remaining_area == 0:
                                        break
                total_profit += year_profit
                print(f"{year}年的总收益为: {year_profit}元")

    # 转换为DataFrame并返回
    planting_plan_df = pd.DataFrame(planting_plan)
    # 输出总收益
    print(f"2024-2030年的总种植收益为: {total_profit}元")
    return planting_plan_df, total_profit

# 示例调用
# 调用该函数处理ABC和DEF类土地的分配
planting_plan_ABC_DEF, total_profit = greedy_crop_strategy_ABC_DEF(land_info, land_crop_efficiency_rank, crop_total_yield)

# 打印种植策略并保存到Excel文件
output_file_strategy = "ABC_DEF种植策略.xlsx"
planting_plan_ABC_DEF.to_excel(output_file_strategy, index=False)
print(f"2024-2030年的总收益为: {total_profit}元")
print(f"种植策略已保存至{output_file_strategy}")