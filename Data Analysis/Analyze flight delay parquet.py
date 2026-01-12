"""
航班延误数据全面分析脚本 (2018-2022)

目标：
1. 了解数据基本情况
2. 分析延误原因分布
3. 发现延误模式和趋势
4. 为特征工程提供insights

运行环境：PySpark
数据：BTS 2018-2022年航班数据
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================================
# 1. 初始化Spark
# ============================================================================

print("=" * 80)
print("航班延误数据全面分析")
print("=" * 80)

spark = SparkSession.builder \
    .appName("Flight Delay Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

print("\n✓ Spark初始化完成")

# ============================================================================
# 2. 读取数据 - 修正版
# ============================================================================

print("\n" + "=" * 80)
print("读取数据")
print("=" * 80)

# 修正后的数据路径 - 使用Parquet格式
BASE_PATH = "/Users/jorahmormont/PycharmProjects/BigDataFinalProject/datasets/robikscube/flight-delay-dataset-20182022/versions/4/"

print("正在加载Parquet文件...")

# 读取所有年份的Parquet文件
flight_df = spark.read.parquet(
    f"{BASE_PATH}Combined_Flights_2018.parquet",
    f"{BASE_PATH}Combined_Flights_2019.parquet",
    f"{BASE_PATH}Combined_Flights_2020.parquet",
    f"{BASE_PATH}Combined_Flights_2021.parquet",
    f"{BASE_PATH}Combined_Flights_2022.parquet"
)

print(f"\n✓ 数据加载完成")
print(f"  总记录数: {flight_df.count():,}")
print(f"  总列数: {len(flight_df.columns)}")

# 显示前几行
print("\n数据预览:")
flight_df.show(5, truncate=False)

# ============================================================================
# 3. 基本数据质量检查
# ============================================================================

print("\n" + "=" * 80)
print("数据质量检查")
print("=" * 80)

# 检查缺失值
print("\n【关键字段缺失值统计】")
key_columns = [
    'Year', 'Month', 'DayofMonth', 'DayOfWeek',
    'Origin', 'Dest', 'Marketing_Airline_Network',
    'DepDelay', 'ArrDelay', 'DepDel15', 'ArrDel15',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    'Cancelled', 'Diverted'
]

missing_stats = []
for col in key_columns:
    if col in flight_df.columns:
        total = flight_df.count()
        missing = flight_df.filter(F.col(col).isNull()).count()
        pct = (missing / total) * 100
        missing_stats.append({
            'column': col,
            'missing_count': missing,
            'missing_pct': round(pct, 2)
        })

missing_df = pd.DataFrame(missing_stats)
print(missing_df.to_string(index=False))

# 数据时间范围
print("\n【数据时间范围】")
flight_df.select(
    F.min('Year').alias('min_year'),
    F.max('Year').alias('max_year'),
    F.min('Month').alias('min_month'),
    F.max('Month').alias('max_month')
).show()

# ============================================================================
# 4. 整体延误统计
# ============================================================================

print("\n" + "=" * 80)
print("整体延误统计")
print("=" * 80)

# 延误定义：DepDel15 = 1 表示延误超过15分钟
# 只统计未取消的航班 - 修正：Cancelled是布尔类型
active_flights = flight_df.filter(F.col('Cancelled') == False)

print("\n【总体延误率】")
overall_stats = active_flights.agg(
    F.count('*').alias('total_flights'),
    F.sum('DepDel15').alias('delayed_flights'),
    (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct'),
    F.avg('DepDelayMinutes').alias('avg_delay_minutes'),
    F.sum(F.col('Cancelled').cast('int')).alias('cancelled_flights')  # 布尔转整数求和
)
overall_stats.show()

# 按年份统计
print("\n【按年份统计】")
yearly_stats = active_flights.groupBy('Year') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct'),
        F.avg('DepDelayMinutes').alias('avg_delay_minutes')
    ) \
    .orderBy('Year')
yearly_stats.show()

# 按月份统计
print("\n【按月份统计】")
monthly_stats = active_flights.groupBy('Month') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .orderBy('Month')
monthly_stats.show(12)

# 按星期几统计
print("\n【按星期几统计】")
dow_stats = active_flights.groupBy('DayOfWeek') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .orderBy('DayOfWeek')
dow_stats.show()

# ============================================================================
# 5. 延误原因分析 ⭐⭐⭐ 核心部分
# ============================================================================

print("\n" + "=" * 80)
print("延误原因深度分析 ⭐")
print("=" * 80)

# 只分析实际延误的航班（DepDel15 = 1）- 修正：Cancelled是布尔类型
delayed_flights = flight_df.filter(
    (F.col('DepDel15') == 1) & (F.col('Cancelled') == False)
)

print(f"\n延误航班总数: {delayed_flights.count():,}")

# 检查延误原因列是否存在
delay_cause_columns = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
available_delay_columns = [col for col in delay_cause_columns if col in flight_df.columns]

if not available_delay_columns:
    print("❌ 未找到延误原因字段，跳过延误原因分析")
else:
    print(f"可用的延误原因字段: {available_delay_columns}")

    # 注意：延误原因字段只有在延误超过15分钟时才有值
    print("\n【延误原因统计】")
    print("注意：只有延误≥15分钟的航班才有原因分类")

    # 构建聚合表达式
    agg_exprs = []
    for col_name in available_delay_columns:
        agg_exprs.extend([
            F.sum(col_name).alias(f'{col_name.lower()}_minutes'),
            F.sum(F.when(F.col(col_name) > 0, 1).otherwise(0)).alias(f'{col_name.lower()}_count')
        ])

    delay_reasons = delayed_flights.agg(*agg_exprs).collect()[0]

    # 转换为易读格式
    delay_analysis_data = []
    friendly_names = {
        'CarrierDelay': 'Carrier (航空公司)',
        'WeatherDelay': 'Weather (天气)',
        'NASDelay': 'NAS (空管系统)',
        'SecurityDelay': 'Security (安全)',
        'LateAircraftDelay': 'Late Aircraft (前序航班)'
    }

    for col_name in available_delay_columns:
        minutes_key = f'{col_name.lower()}_minutes'
        count_key = f'{col_name.lower()}_count'

        delay_analysis_data.append({
            '延误原因': friendly_names.get(col_name, col_name),
            '总延误分钟数': delay_reasons[minutes_key] if delay_reasons[minutes_key] is not None else 0,
            '受影响航班数': delay_reasons[count_key] if delay_reasons[count_key] is not None else 0
        })

    delay_analysis = pd.DataFrame(delay_analysis_data)

    # 计算百分比
    if len(delay_analysis) > 0:
        total_minutes = delay_analysis['总延误分钟数'].sum()
        total_flights_delayed = delay_analysis['受影响航班数'].sum()

        if total_minutes > 0:
            delay_analysis['分钟数占比%'] = (
                delay_analysis['总延误分钟数'] / total_minutes * 100
            ).round(2)

            delay_analysis['航班数占比%'] = (
                delay_analysis['受影响航班数'] / total_flights_delayed * 100
            ).round(2)

            # 计算平均延误时长
            delay_analysis['平均延误(分钟)'] = (
                delay_analysis['总延误分钟数'] / delay_analysis['受影响航班数']
            ).round(2)

            # 按总延误分钟数排序
            delay_analysis = delay_analysis.sort_values('总延误分钟数', ascending=False)

            print("\n" + delay_analysis.to_string(index=False))

            print("\n【关键发现】")
            print(f"  1. 主要延误原因（按分钟数）: {delay_analysis.iloc[0]['延误原因']}")
            print(f"     占比: {delay_analysis.iloc[0]['分钟数占比%']:.1f}%")

            if len(delay_analysis) > 1:
                print(f"  2. 次要延误原因: {delay_analysis.iloc[1]['延误原因']}")
                print(f"     占比: {delay_analysis.iloc[1]['分钟数占比%']:.1f}%")
                print(f"  3. 前两大原因合计: {delay_analysis.iloc[0]['分钟数占比%'] + delay_analysis.iloc[1]['分钟数占比%']:.1f}%")

# ============================================================================
# 6. 延误原因按年份趋势
# ============================================================================

if available_delay_columns:
    print("\n【延误原因随时间变化趋势】")

    # 构建年度延误原因聚合
    yearly_agg_exprs = [F.sum(col).alias(col) for col in available_delay_columns]
    yearly_reasons = delayed_flights.groupBy('Year') \
        .agg(*yearly_agg_exprs) \
        .orderBy('Year')

    yearly_reasons.show()

# ============================================================================
# 7. 航空公司延误分析
# ============================================================================

print("\n" + "=" * 80)
print("航空公司延误分析")
print("=" * 80)

print("\n【前20大航空公司延误率】")
airline_stats = active_flights.groupBy('Marketing_Airline_Network') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct'),
        F.avg('DepDelayMinutes').alias('avg_delay_minutes')
    ) \
    .filter(F.col('total_flights') >= 10000) \
    .orderBy(F.desc('total_flights')) \
    .limit(20)

airline_stats.show(20, truncate=False)

# ============================================================================
# 8. 机场延误分析
# ============================================================================

print("\n" + "=" * 80)
print("机场延误分析")
print("=" * 80)

print("\n【出发延误率最高的前20个机场】")
origin_stats = active_flights.groupBy('Origin') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .filter(F.col('total_flights') >= 10000) \
    .orderBy(F.desc('delay_rate_pct')) \
    .limit(20)

origin_stats.show(20, truncate=False)

print("\n【航班量最大的前20个机场及延误率】")
busiest_airports = active_flights.groupBy('Origin') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .orderBy(F.desc('total_flights')) \
    .limit(20)

busiest_airports.show(20, truncate=False)

# ============================================================================
# 9. 航线延误分析
# ============================================================================

print("\n" + "=" * 80)
print("航线延误分析")
print("=" * 80)

# 创建航线字段
flight_with_route = active_flights.withColumn(
    'route',
    F.concat(F.col('Origin'), F.lit('-'), F.col('Dest'))
)

print("\n【延误率最高的前20条航线（航班数≥1000）】")
route_stats = flight_with_route.groupBy('route') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .filter(F.col('total_flights') >= 1000) \
    .orderBy(F.desc('delay_rate_pct')) \
    .limit(20)

route_stats.show(20, truncate=False)

# ============================================================================
# 10. 时间模式分析
# ============================================================================

print("\n" + "=" * 80)
print("时间模式分析")
print("=" * 80)

# 按小时统计（如果有CRSDepTime字段）
if 'CRSDepTime' in flight_df.columns:
    print("\n【按计划出发时间统计延误率】")

    flight_with_hour = active_flights.withColumn(
        'dep_hour',
        (F.col('CRSDepTime') / 100).cast('int')
    )

    hourly_stats = flight_with_hour.groupBy('dep_hour') \
        .agg(
            F.count('*').alias('total_flights'),
            F.sum('DepDel15').alias('delayed_flights'),
            (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
        ) \
        .orderBy('dep_hour')

    hourly_stats.show(24)

# 按季度统计
print("\n【按季度统计】")
quarterly_stats = active_flights.groupBy('Year', 'Quarter') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
    ) \
    .orderBy('Year', 'Quarter')

quarterly_stats.show(20)

# ============================================================================
# 11. 延误传播分析（如果有Tail_Number）
# ============================================================================

if 'Tail_Number' in flight_df.columns:
    print("\n" + "=" * 80)
    print("延误传播分析")
    print("=" * 80)

    print("\n【同一架飞机的延误传播】")
    print("分析：如果前一个航班延误，下一个航班延误概率")

    # 注意：这里需要FlightDate和CRSDepTime字段
    if all(col in flight_df.columns for col in ['FlightDate', 'CRSDepTime']):
        # 创建窗口按Tail_Number和时间排序
        window_spec = Window.partitionBy('Tail_Number').orderBy('FlightDate', 'CRSDepTime')

        # 获取前一个航班的延误信息
        flight_with_prev = active_flights.withColumn(
            'prev_arr_delay',
            F.lag('ArrDelay').over(window_spec)
        )

        # 分析前序航班延误对当前航班的影响
        propagation_stats = flight_with_prev.filter(
            F.col('prev_arr_delay').isNotNull()
        ).groupBy(
            (F.col('prev_arr_delay') > 15).alias('prev_delayed')
        ).agg(
            F.count('*').alias('total_flights'),
            F.sum('DepDel15').alias('delayed_flights'),
            (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct')
        )

        propagation_stats.show()
    else:
        print("缺少FlightDate或CRSDepTime字段，无法进行延误传播分析")

# ============================================================================
# 12. COVID-19 影响分析
# ============================================================================

print("\n" + "=" * 80)
print("COVID-19 影响分析")
print("=" * 80)

print("\n【2020年（COVID年）vs 其他年份对比】")

covid_comparison = active_flights.withColumn(
    'period',
    F.when(F.col('Year') == 2020, 'COVID-2020')
     .when(F.col('Year') < 2020, 'Pre-COVID (2018-2019)')
     .otherwise('Post-COVID (2021-2022)')
).groupBy('period') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum('DepDel15').alias('delayed_flights'),
        (F.sum('DepDel15') / F.count('*') * 100).alias('delay_rate_pct'),
        F.avg('DepDelayMinutes').alias('avg_delay_minutes')
    )

covid_comparison.show(truncate=False)

# ============================================================================
# 13. 延误严重程度分布
# ============================================================================

print("\n" + "=" * 80)
print("延误严重程度分布")
print("=" * 80)

print("\n【延误时长分布】")

delay_severity = delayed_flights.withColumn(
    'delay_category',
    F.when(F.col('DepDelayMinutes').between(15, 30), '15-30分钟')
     .when(F.col('DepDelayMinutes').between(31, 60), '31-60分钟')
     .when(F.col('DepDelayMinutes').between(61, 120), '1-2小时')
     .when(F.col('DepDelayMinutes').between(121, 240), '2-4小时')
     .when(F.col('DepDelayMinutes') > 240, '>4小时')
     .otherwise('其他')
).groupBy('delay_category') \
    .agg(
        F.count('*').alias('flight_count'),
        (F.count('*') / delayed_flights.count() * 100).alias('percentage')
    ) \
    .orderBy('flight_count', ascending=False)

delay_severity.show()

# ============================================================================
# 14. 取消和改航分析 - 修正版
# ============================================================================

print("\n" + "=" * 80)
print("取消和改航分析")
print("=" * 80)

print("\n【取消和改航统计】")

cancellation_stats = flight_df.agg(
    F.count('*').alias('total_flights'),
    F.sum(F.col('Cancelled').cast('int')).alias('cancelled_flights'),  # 布尔转整数
    (F.sum(F.col('Cancelled').cast('int')) / F.count('*') * 100).alias('cancellation_rate_pct'),
    F.sum(F.col('Diverted').cast('int')).alias('diverted_flights'),  # 修正：Diverted也需要类型转换
    (F.sum(F.col('Diverted').cast('int')) / F.count('*') * 100).alias('diversion_rate_pct')
)

cancellation_stats.show()

# 按年份
print("\n【取消率按年份】")
yearly_cancellation = flight_df.groupBy('Year') \
    .agg(
        F.count('*').alias('total_flights'),
        F.sum(F.col('Cancelled').cast('int')).alias('cancelled'),  # 布尔转整数
        (F.sum(F.col('Cancelled').cast('int')) / F.count('*') * 100).alias('cancel_rate_pct')
    ) \
    .orderBy('Year')

yearly_cancellation.show()

# ============================================================================
# 15. 保存分析结果
# ============================================================================

print("\n" + "=" * 80)
print("保存分析结果")
print("=" * 80)

# 将关键统计结果转为Pandas并保存
print("\n保存分析报告...")

# 1. 年度趋势
yearly_stats_pd = yearly_stats.toPandas()
yearly_stats_pd.to_csv('analysis_yearly_trends.csv', index=False, encoding='utf-8-sig')
print("✓ 年度趋势: analysis_yearly_trends.csv")

# 2. 航空公司排名
airline_stats_pd = airline_stats.toPandas()
airline_stats_pd.to_csv('analysis_airline_ranking.csv', index=False, encoding='utf-8-sig')
print("✓ 航空公司排名: analysis_airline_ranking.csv")

# 3. 机场排名
busiest_airports_pd = busiest_airports.toPandas()
busiest_airports_pd.to_csv('analysis_airport_ranking.csv', index=False, encoding='utf-8-sig')
print("✓ 机场排名: analysis_airport_ranking.csv")

# 4. 延误原因分析（如果存在）
if 'delay_analysis' in locals() and len(delay_analysis) > 0:
    delay_analysis.to_csv('analysis_delay_reasons.csv', index=False, encoding='utf-8-sig')
    print("✓ 延误原因分析: analysis_delay_reasons.csv")

# 5. 月度趋势
monthly_stats_pd = monthly_stats.toPandas()
monthly_stats_pd.to_csv('analysis_monthly_trends.csv', index=False, encoding='utf-8-sig')
print("✓ 月度趋势: analysis_monthly_trends.csv")

# ============================================================================
# 16. 总结和关键发现
# ============================================================================

print("\n" + "=" * 80)
print("关键发现总结")
print("=" * 80)

if 'delay_analysis' in locals() and len(delay_analysis) > 0:
    print("\n【延误原因 Top 3】")
    for i in range(min(3, len(delay_analysis))):
        print(f"  {i+1}. {delay_analysis.iloc[i]['延误原因']}: "
              f"{delay_analysis.iloc[i]['分钟数占比%']:.1f}% "
              f"({delay_analysis.iloc[i]['受影响航班数']:,} 航班)")
else:
    print("\n【延误原因】: 数据不可用")

print("\n【数据质量评估】")
if len(missing_stats) > 0:
    avg_missing_pct = missing_df['missing_pct'].mean()
    print(f"  • 数据完整性: {100 - avg_missing_pct:.1f}%")
else:
    print("  • 数据完整性: 无法计算")

print("\n" + "=" * 80)
print("✅ 分析完成！")
print("=" * 80)

# 停止Spark
spark.stop()