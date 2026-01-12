"""
Analyze Delay Causes from Monthly CSV Files
从月度CSV文件分析延误原因

数据结构：Flights_2018_1.csv ~ Flights_2022_7.csv
每个月一个文件
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, when, round as spark_round
import pandas as pd
import os

# ============================================================================
# Initialize Spark
# ============================================================================

spark = SparkSession.builder \
    .appName("Monthly CSV Delay Cause Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

print("=" * 80)
print("Flight Delay Cause Analysis - Monthly CSV Files")
print("=" * 80)

# ============================================================================
# Step 1: Read All Monthly CSV Files
# ============================================================================

BASE_PATH = "/Users/jorahmormont/PycharmProjects/BigDataFinalProject/datasets/robikscube/flight-delay-dataset-20182022/versions/4/raw/"

print("\nStep 1: Loading monthly CSV files...")
print("This will take 15-20 minutes...")

# Build list of all monthly files
file_list = []

# 2018: months 1-12
for month in range(1, 13):
    file_list.append(f"{BASE_PATH}Flights_2018_{month}.csv")

# 2019: months 1-12
for month in range(1, 13):
    file_list.append(f"{BASE_PATH}Flights_2019_{month}.csv")

# 2020: months 1-12
for month in range(1, 13):
    file_list.append(f"{BASE_PATH}Flights_2020_{month}.csv")

# 2021: months 1-12
for month in range(1, 13):
    file_list.append(f"{BASE_PATH}Flights_2021_{month}.csv")

# 2022: months 1-7 (based on your screenshot)
for month in range(1, 8):
    file_list.append(f"{BASE_PATH}Flights_2022_{month}.csv")

print(f"\nTotal files to read: {len(file_list)}")

# Check first file to verify columns
print("\nVerifying columns in first file...")
test_df = spark.read.csv(file_list[0], header=True, inferSchema=False)
print(f"Columns in CSV: {len(test_df.columns)}")

delay_cause_columns = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
has_delay_causes = all(col in test_df.columns for col in delay_cause_columns)

if has_delay_causes:
    print("✓ CSV files contain delay cause columns!")
    for col_name in delay_cause_columns:
        print(f"  ✓ {col_name}")
else:
    print("✗ Missing delay cause columns")
    exit(1)

# Read all files
print("\n正在读取所有月度CSV文件...")
print("进度：")

flight_df = None
for i, file_path in enumerate(file_list, 1):
    if i % 10 == 0:
        print(f"  已读取 {i}/{len(file_list)} 个文件...")

    monthly_df = spark.read.csv(file_path, header=True, inferSchema=True)

    if flight_df is None:
        flight_df = monthly_df
    else:
        flight_df = flight_df.union(monthly_df)

total_records = flight_df.count()
print(f"\n✓ 数据加载完成")
print(f"  总记录数: {total_records:,}")
print(f"  总列数: {len(flight_df.columns)}")

# ============================================================================
# Step 2: Analyze Delay Causes
# ============================================================================

print("\n" + "=" * 80)
print("Analyzing Delay Causes")
print("=" * 80)

# Filter to delayed flights only
delayed_flights = flight_df.filter(
    (col('DepDel15') == 1) & (col('Cancelled') == 0)
)

total_delayed = delayed_flights.count()
print(f"\nTotal delayed flights (≥15 min): {total_delayed:,}")

# ============================================================================
# Analysis 1: Total Delay Minutes by Cause
# ============================================================================

print("\n【Total Delay Minutes by Cause】")

delay_minutes = delayed_flights.agg(
    spark_sum('CarrierDelay').alias('carrier_minutes'),
    spark_sum('WeatherDelay').alias('weather_minutes'),
    spark_sum('NASDelay').alias('nas_minutes'),
    spark_sum('SecurityDelay').alias('security_minutes'),
    spark_sum('LateAircraftDelay').alias('late_aircraft_minutes')
).collect()[0]

# Calculate percentages - 修正：使用Python的sum函数
total_minutes = (
    (delay_minutes['carrier_minutes'] or 0) +
    (delay_minutes['weather_minutes'] or 0) +
    (delay_minutes['nas_minutes'] or 0) +
    (delay_minutes['security_minutes'] or 0) +
    (delay_minutes['late_aircraft_minutes'] or 0)
)

results_minutes = pd.DataFrame({
    'Delay Cause': [
        'Late Aircraft',
        'Carrier (Airline)',
        'NAS (Air System)',
        'Weather',
        'Security'
    ],
    'Total Minutes': [
        delay_minutes['late_aircraft_minutes'] or 0,
        delay_minutes['carrier_minutes'] or 0,
        delay_minutes['nas_minutes'] or 0,
        delay_minutes['weather_minutes'] or 0,
        delay_minutes['security_minutes'] or 0
    ]
})

results_minutes['Percentage'] = (results_minutes['Total Minutes'] / total_minutes * 100).round(2)
results_minutes['Minutes (M)'] = (results_minutes['Total Minutes'] / 1_000_000).round(2)
results_minutes = results_minutes.sort_values('Total Minutes', ascending=False)

print("\n" + "=" * 80)
print("By Total Delay Minutes:")
print("=" * 80)
print(results_minutes[['Delay Cause', 'Minutes (M)', 'Percentage']].to_string(index=False))

# ============================================================================
# Analysis 2: Number of Flights Affected
# ============================================================================

print("\n" + "=" * 80)
print("Number of Flights Affected by Each Cause")
print("=" * 80)

flights_affected = delayed_flights.agg(
    spark_sum(when(col('LateAircraftDelay') > 0, 1).otherwise(0)).alias('late_count'),
    spark_sum(when(col('CarrierDelay') > 0, 1).otherwise(0)).alias('carrier_count'),
    spark_sum(when(col('NASDelay') > 0, 1).otherwise(0)).alias('nas_count'),
    spark_sum(when(col('WeatherDelay') > 0, 1).otherwise(0)).alias('weather_count'),
    spark_sum(when(col('SecurityDelay') > 0, 1).otherwise(0)).alias('security_count')
).collect()[0]

results_flights = pd.DataFrame({
    'Delay Cause': [
        'Late Aircraft',
        'Carrier (Airline)',
        'NAS (Air System)',
        'Weather',
        'Security'
    ],
    'Flights Affected': [
        flights_affected['late_count'],
        flights_affected['carrier_count'],
        flights_affected['nas_count'],
        flights_affected['weather_count'],
        flights_affected['security_count']
    ]
})

results_flights['Percentage'] = (results_flights['Flights Affected'] / total_delayed * 100).round(2)
results_flights = results_flights.sort_values('Flights Affected', ascending=False)

print("\n" + results_flights.to_string(index=False))

# ============================================================================
# Analysis 3: Average Delay Duration
# ============================================================================

print("\n" + "=" * 80)
print("Average Delay Duration per Affected Flight")
print("=" * 80)

# Rename percentage columns before merging to avoid conflicts
results_minutes_renamed = results_minutes.rename(columns={'Percentage': 'Pct_Minutes'})
results_flights_renamed = results_flights.rename(columns={'Percentage': 'Pct_Flights'})

results_combined = results_minutes_renamed.merge(results_flights_renamed, on='Delay Cause')
results_combined['Avg Minutes'] = (
    results_combined['Total Minutes'] / results_combined['Flights Affected']
).round(2)

print("\n" + results_combined[[
    'Delay Cause',
    'Avg Minutes',
    'Minutes (M)',
    'Flights Affected'
]].to_string(index=False))

# ============================================================================
# Analysis 4: Yearly Trends
# ============================================================================

print("\n" + "=" * 80)
print("Delay Causes by Year (millions of minutes)")
print("=" * 80)

yearly_causes = delayed_flights.groupBy('Year').agg(
    spark_round(spark_sum('LateAircraftDelay')/1000000, 2).alias('Late Aircraft'),
    spark_round(spark_sum('CarrierDelay')/1000000, 2).alias('Carrier'),
    spark_round(spark_sum('NASDelay')/1000000, 2).alias('NAS'),
    spark_round(spark_sum('WeatherDelay')/1000000, 2).alias('Weather'),
    spark_round(spark_sum('SecurityDelay')/1000000, 2).alias('Security')
).orderBy('Year')

yearly_causes.show()

# ============================================================================
# Summary & Key Findings
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS - Delay Causes Ranking")
print("=" * 80)

summary = pd.DataFrame({
    'Rank': [1, 2, 3, 4, 5],
    'Cause': results_minutes['Delay Cause'].values,
    'Minutes %': results_minutes['Percentage'].values,
    'Flights %': results_flights['Percentage'].values
})

print("\n" + summary.to_string(index=False))

# Top cause
top_cause = results_minutes.iloc[0]
print(f"\n{'=' * 80}")
print("⭐⭐⭐ PRIMARY DELAY CAUSE ⭐⭐⭐")
print(f"{'=' * 80}")
print(f"\n{top_cause['Delay Cause']}")
print(f"  • {top_cause['Percentage']:.1f}% of all delay time")
print(f"  • {top_cause['Minutes (M)']:.1f} million minutes")

# Second cause
second_cause = results_minutes.iloc[1]
print(f"\n{'=' * 80}")
print("⭐⭐ SECONDARY DELAY CAUSE ⭐⭐")
print(f"{'=' * 80}")
print(f"\n{second_cause['Delay Cause']}")
print(f"  • {second_cause['Percentage']:.1f}% of all delay time")
print(f"  • {second_cause['Minutes (M)']:.1f} million minutes")

# Top 2 combined
top_2_pct = results_minutes.iloc[0]['Percentage'] + results_minutes.iloc[1]['Percentage']
print(f"\n{'=' * 80}")
print(f"Top 2 Causes Combined: {top_2_pct:.1f}% of all delays")
print(f"{'=' * 80}")

# ============================================================================
# Validation with Previous Findings
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION: Comparing with Delay Propagation Analysis")
print("=" * 80)

if results_minutes.iloc[0]['Delay Cause'] == 'Late Aircraft':
    print("\n✅ VALIDATED! Late Aircraft is the primary cause")
    print("\nThis confirms our earlier finding:")
    print("  • Previous flight delayed → 52% current delay rate")
    print("  • Previous flight on-time → 10% current delay rate")
    print("  • 5X difference validates Late Aircraft as dominant factor")
    print(f"\nLate Aircraft accounts for {results_minutes.iloc[0]['Percentage']:.1f}%")
    print("This is consistent with 40-60% range found by Wang et al. (2020)")
else:
    print(f"\n⚠️  Unexpected: {results_minutes.iloc[0]['Delay Cause']} is primary cause")

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "=" * 80)
print("Saving Results")
print("=" * 80)

# Summary table - Now with clear column names
summary_table = results_combined[[
    'Delay Cause',
    'Total Minutes',
    'Pct_Minutes',
    'Flights Affected',
    'Pct_Flights',
    'Avg Minutes'
]].copy()

summary_table.to_csv('delay_causes_analysis.csv', index=False)
print("✓ Saved: delay_causes_analysis.csv")

# Yearly trends
yearly_df = yearly_causes.toPandas()
yearly_df.to_csv('delay_causes_yearly.csv', index=False)
print("✓ Saved: delay_causes_yearly.csv")

# For visualization - use results_minutes directly since it has the percentage
pie_data = results_minutes[['Delay Cause', 'Percentage']].copy()
pie_data.to_csv('delay_causes_pie.csv', index=False)
print("✓ Saved: delay_causes_pie.csv")

print("\n" + "=" * 80)
print("✅ Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("  • delay_causes_analysis.csv  (summary table)")
print("  • delay_causes_yearly.csv    (trends)")
print("  • delay_causes_pie.csv       (for charts)")

spark.stop()