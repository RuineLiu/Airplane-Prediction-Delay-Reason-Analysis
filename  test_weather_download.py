#!/usr/bin/env python3
"""
NOAA天气数据获取 - 快速测试脚本
用于验证数据获取是否正常工作
"""

print("正在测试NOAA天气数据获取...")
print("=" * 70)

# 首先检查是否安装了必要的库
print("\n1. 检查依赖库...")
try:
    import pandas as pd

    print("   ✓ pandas 已安装")
except ImportError:
    print("   ✗ pandas 未安装，请运行: pip install pandas")
    exit(1)

try:
    from meteostat import Stations, Hourly

    print("   ✓ meteostat 已安装")
except ImportError:
    print("   ✗ meteostat 未安装，请运行: pip install meteostat")
    exit(1)

from datetime import datetime

print("\n2. 测试：查找JFK机场附近的气象站...")
try:
    # JFK机场坐标
    jfk_lat, jfk_lon = 40.6413, -73.7781

    stations = Stations()
    stations = stations.nearby(jfk_lat, jfk_lon)
    station = stations.fetch(5)  # 获取最近的5个气象站

    if not station.empty:
        print(f"   ✓ 找到 {len(station)} 个气象站")
        print("\n   最近的气象站信息：")
        print(station[['name', 'distance']].to_string())

        # 使用最近的气象站
        nearest_station_id = station.index[0]
        print(f"\n   将使用气象站: {nearest_station_id}")

        print("\n3. 测试：下载2018年1月的天气数据...")
        start = datetime(2018, 1, 1)
        end = datetime(2018, 1, 7)  # 只下载一周数据用于测试

        data = Hourly(nearest_station_id, start, end)
        weather_df = data.fetch()

        if not weather_df.empty:
            print(f"   ✓ 成功获取 {len(weather_df)} 条记录")
            print("\n   数据预览：")
            print(weather_df.head())

            print("\n   可用字段：")
            print(f"   {', '.join(weather_df.columns.tolist())}")

            print("\n   数据统计：")
            print(weather_df.describe())

            # 保存示例数据
            weather_df.to_csv('jfk_weather_sample.csv')
            print("\n   ✓ 示例数据已保存到: jfk_weather_sample.csv")

            print("\n" + "=" * 70)
            print("✓ 测试成功！meteostat库工作正常")
            print("=" * 70)
            print("\n下一步：")
            print("1. 准备你的所有机场列表（机场代码+经纬度）")
            print("2. 使用完整工作流脚本批量下载所有数据")
            print("3. 参考 noaa_weather_data_guide.py 中的完整示例")

        else:
            print("   ✗ 未获取到天气数据")
    else:
        print("   ✗ 未找到气象站")

except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    print("\n可能的原因：")
    print("1. 网络连接问题")
    print("2. meteostat服务器暂时不可用")
    print("3. 需要更新meteostat库: pip install --upgrade meteostat")