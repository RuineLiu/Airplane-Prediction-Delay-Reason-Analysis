"""
节假日数据生成脚本
用于航班延误预测项目

生成2018-2024年美国节假日数据
包含特征工程所需的所有字段
"""

import holidays
import pandas as pd
from datetime import datetime, timedelta

print("=" * 80)
print("航班延误预测项目 - 节假日数据生成器")
print("=" * 80)

# ============================================================================
# 1. 生成美国联邦节假日 (2018-2024)
# ============================================================================

print("\n步骤 1: 生成美国联邦节假日...")

# 生成2018-2024年的所有联邦节假日
us_holidays = holidays.US(years=range(2018, 2025))

print(f"✓ 找到 {len(us_holidays)} 个联邦节假日")

# ============================================================================
# 2. 创建详细的节假日数据集
# ============================================================================

print("\n步骤 2: 创建详细数据集...")

holiday_data = []

for date, name in us_holidays.items():

    # 判断是否为重大节假日（影响航班较大的）
    is_major = name in [
        'Thanksgiving',  # 感恩节（最重要）
        'Christmas Day',  # 圣诞节（最重要）
        "New Year's Day",  # 新年（最重要）
        'Independence Day',  # 独立日（重要）
        'Memorial Day',  # 阵亡将士纪念日（重要）
        'Labor Day',  # 劳动节（重要）
        "New Year's Day (Observed)",
        'Christmas Day (Observed)'
    ]

    # 节假日分类
    if name in ['Thanksgiving', 'Christmas Day', "New Year's Day",
                'Christmas Day (Observed)', "New Year's Day (Observed)"]:
        category = 'Winter Holiday'
        impact_level = 'extreme'  # 极高影响
    elif name in ['Memorial Day', 'Independence Day', 'Labor Day']:
        category = 'Summer Holiday'
        impact_level = 'high'  # 高影响
    else:
        category = 'Other Holiday'
        impact_level = 'medium'  # 中等影响

    # 添加数据
    holiday_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'holiday_name': name,
        'is_major_holiday': 1 if is_major else 0,
        'holiday_category': category,
        'impact_level': impact_level,
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'day_of_week': date.strftime('%A'),
        'day_of_week_num': date.weekday() + 1  # 1=Monday, 7=Sunday
    })

# 转换为DataFrame
holiday_df = pd.DataFrame(holiday_data)
holiday_df = holiday_df.sort_values('date').reset_index(drop=True)

print(f"✓ 生成了 {len(holiday_df)} 条节假日记录")

# ============================================================================
# 3. 添加学校假期和特殊时期
# ============================================================================

print("\n步骤 3: 添加学校假期和特殊旅游时期...")

# 学校假期数据（美国主要学校假期）
school_breaks = []

for year in range(2018, 2025):

    # 春假 (Spring Break) - 通常在3月中旬
    spring_break_start = datetime(year, 3, 10)
    for i in range(10):  # 10天春假
        spring_break_start += timedelta(days=1)
        school_breaks.append({
            'date': spring_break_start.strftime('%Y-%m-%d'),
            'holiday_name': 'Spring Break',
            'is_major_holiday': 0,
            'holiday_category': 'School Break',
            'impact_level': 'high',
            'year': year,
            'month': 3,
            'day': spring_break_start.day,
            'day_of_week': spring_break_start.strftime('%A'),
            'day_of_week_num': spring_break_start.weekday() + 1
        })

    # 暑假高峰期标记 (Summer Peak) - 6月中旬到8月中旬
    summer_start = datetime(year, 6, 15)
    summer_end = datetime(year, 8, 15)
    current = summer_start
    while current <= summer_end:
        school_breaks.append({
            'date': current.strftime('%Y-%m-%d'),
            'holiday_name': 'Summer Vacation Peak',
            'is_major_holiday': 0,
            'holiday_category': 'Summer Peak',
            'impact_level': 'medium',
            'year': year,
            'month': current.month,
            'day': current.day,
            'day_of_week': current.strftime('%A'),
            'day_of_week_num': current.weekday() + 1
        })
        current += timedelta(days=7)  # 每周标记一次

    # 感恩节周 (Thanksgiving Week) - 感恩节前后3天
    thanksgiving = datetime(year, 11, 1)
    # 找到11月的第4个星期四
    while thanksgiving.weekday() != 3:  # 3 = Thursday
        thanksgiving += timedelta(days=1)
    thanksgiving += timedelta(weeks=3)

    for offset in [-3, -2, -1, 1, 2]:  # 前3天和后2天
        date = thanksgiving + timedelta(days=offset)
        if date.strftime('%Y-%m-%d') not in holiday_df['date'].values:
            school_breaks.append({
                'date': date.strftime('%Y-%m-%d'),
                'holiday_name': 'Thanksgiving Week',
                'is_major_holiday': 1,
                'holiday_category': 'Winter Holiday',
                'impact_level': 'extreme',
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.strftime('%A'),
                'day_of_week_num': date.weekday() + 1
            })

    # 圣诞周 (Christmas Week) - 圣诞节前后
    christmas = datetime(year, 12, 25)
    for offset in range(-3, 4):  # 前后3天
        date = christmas + timedelta(days=offset)
        if date.strftime('%Y-%m-%d') not in holiday_df['date'].values:
            school_breaks.append({
                'date': date.strftime('%Y-%m-%d'),
                'holiday_name': 'Christmas Week',
                'is_major_holiday': 1,
                'holiday_category': 'Winter Holiday',
                'impact_level': 'extreme',
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.strftime('%A'),
                'day_of_week_num': date.weekday() + 1
            })

# 合并学校假期数据
school_breaks_df = pd.DataFrame(school_breaks)
holiday_df_extended = pd.concat([holiday_df, school_breaks_df], ignore_index=True)
holiday_df_extended = holiday_df_extended.sort_values('date').reset_index(drop=True)

print(f"✓ 添加了 {len(school_breaks)} 个学校假期/特殊时期标记")
print(f"✓ 总共 {len(holiday_df_extended)} 条记录")

# ============================================================================
# 4. 创建节假日前后的特征数据
# ============================================================================

print("\n步骤 4: 创建节假日特征数据（用于PySpark join）...")

# 创建一个完整的日期范围 (2018-01-01 to 2024-12-31)
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start_date, end_date, freq='D')

# 创建完整的日期DataFrame
full_dates_df = pd.DataFrame({
    'date': date_range.strftime('%Y-%m-%d'),
    'year': date_range.year,
    'month': date_range.month,
    'day': date_range.day,
    'day_of_week': date_range.strftime('%A'),
    'day_of_week_num': date_range.weekday + 1
})

# 标记是否是节假日
holiday_dates_set = set(holiday_df_extended['date'].values)
full_dates_df['is_holiday'] = full_dates_df['date'].isin(holiday_dates_set).astype(int)

# 标记是否是重大节假日
major_holiday_dates = set(holiday_df_extended[
                              holiday_df_extended['is_major_holiday'] == 1
                              ]['date'].values)
full_dates_df['is_major_holiday'] = full_dates_df['date'].isin(major_holiday_dates).astype(int)


# 计算距离下一个节假日的天数
def days_to_next_holiday(date_str, holiday_dates):
    current_date = datetime.strptime(date_str, '%Y-%m-%d')
    future_holidays = [
        datetime.strptime(h, '%Y-%m-%d')
        for h in holiday_dates
        if datetime.strptime(h, '%Y-%m-%d') > current_date
    ]
    if future_holidays:
        next_holiday = min(future_holidays)
        return (next_holiday - current_date).days
    return 999  # 如果没有未来节假日，返回大数


# 计算距离上一个节假日的天数
def days_from_prev_holiday(date_str, holiday_dates):
    current_date = datetime.strptime(date_str, '%Y-%m-%d')
    past_holidays = [
        datetime.strptime(h, '%Y-%m-%d')
        for h in holiday_dates
        if datetime.strptime(h, '%Y-%m-%d') < current_date
    ]
    if past_holidays:
        prev_holiday = max(past_holidays)
        return (current_date - prev_holiday).days
    return 999


print("  计算节假日距离特征（这可能需要1-2分钟）...")

full_dates_df['days_to_next_holiday'] = full_dates_df['date'].apply(
    lambda x: days_to_next_holiday(x, holiday_dates_set)
)

full_dates_df['days_from_prev_holiday'] = full_dates_df['date'].apply(
    lambda x: days_from_prev_holiday(x, holiday_dates_set)
)

# 创建节假日窗口特征
full_dates_df['is_holiday_week_before'] = (
        (full_dates_df['days_to_next_holiday'] >= 0) &
        (full_dates_df['days_to_next_holiday'] <= 7)
).astype(int)

full_dates_df['is_holiday_week_after'] = (
        (full_dates_df['days_from_prev_holiday'] >= 0) &
        (full_dates_df['days_from_prev_holiday'] <= 7)
).astype(int)

# 节假日前3天（关键特征！）
full_dates_df['is_3days_before_holiday'] = (
        (full_dates_df['days_to_next_holiday'] >= 0) &
        (full_dates_df['days_to_next_holiday'] <= 3)
).astype(int)

# 周末标记
full_dates_df['is_weekend'] = (full_dates_df['day_of_week_num'] >= 6).astype(int)

print(f"✓ 创建了完整的日期特征数据集：{len(full_dates_df)} 天")

# ============================================================================
# 5. 保存数据
# ============================================================================

print("\n步骤 5: 保存数据文件...")

# 文件1: 基础节假日数据（简洁版）
output_file_1 = 'us_holidays_2018_2024_basic.csv'
holiday_df.to_csv(output_file_1, index=False)
print(f"✓ 已保存: {output_file_1}")
print(f"  内容: 联邦节假日基础信息")

# 文件2: 扩展节假日数据（包含学校假期）
output_file_2 = 'us_holidays_2018_2024_extended.csv'
holiday_df_extended.to_csv(output_file_2, index=False)
print(f"✓ 已保存: {output_file_2}")
print(f"  内容: 联邦节假日 + 学校假期 + 特殊时期")

# 文件3: 完整日期特征数据（用于PySpark join）
output_file_3 = 'holiday_features_2018_2024_complete.csv'
full_dates_df.to_csv(output_file_3, index=False)
print(f"✓ 已保存: {output_file_3}")
print(f"  内容: 每一天的节假日特征（用于直接join）")

# ============================================================================
# 6. 数据统计和预览
# ============================================================================

print("\n" + "=" * 80)
print("数据统计")
print("=" * 80)

print("\n【文件1: 基础节假日数据】")
print(f"  总记录数: {len(holiday_df)}")
print(f"  时间范围: {holiday_df['date'].min()} 到 {holiday_df['date'].max()}")
print(f"\n  按年份统计:")
print(holiday_df.groupby('year').size())
print(f"\n  按类别统计:")
print(holiday_df.groupby('holiday_category').size())
print(f"\n  重大节假日数: {holiday_df['is_major_holiday'].sum()}")

print("\n【文件2: 扩展节假日数据】")
print(f"  总记录数: {len(holiday_df_extended)}")
print(f"\n  按类别统计:")
print(holiday_df_extended.groupby('holiday_category').size())

print("\n【文件3: 完整特征数据】")
print(f"  总天数: {len(full_dates_df)}")
print(f"  节假日天数: {full_dates_df['is_holiday'].sum()}")
print(f"  重大节假日天数: {full_dates_df['is_major_holiday'].sum()}")
print(f"  节假日前一周天数: {full_dates_df['is_holiday_week_before'].sum()}")
print(f"  周末天数: {full_dates_df['is_weekend'].sum()}")

print("\n" + "=" * 80)
print("数据预览")
print("=" * 80)

print("\n【基础节假日数据 - 前10条】")
print(holiday_df.head(10).to_string(index=False))

print("\n【完整特征数据 - 2022年感恩节前后】")
thanksgiving_preview = full_dates_df[
    (full_dates_df['date'] >= '2022-11-20') &
    (full_dates_df['date'] <= '2022-11-28')
    ]
print(thanksgiving_preview.to_string(index=False))

print("\n" + "=" * 80)
print("✅ 节假日数据生成完成！")
print("=" * 80)

print("\n生成的文件:")
print(f"  1. {output_file_1} - 用于数据探索")
print(f"  2. {output_file_2} - 包含完整节假日信息")
print(f"  3. {output_file_3} - ⭐ 推荐用于PySpark join")
