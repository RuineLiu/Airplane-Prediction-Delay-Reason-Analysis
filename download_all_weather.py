import pandas as pd
from meteostat import Stations, Hourly
from datetime import datetime
import time

# 1. 读取机场列表
airports_df = pd.read_csv('us_airports_coordinates.csv')


# 2. 为每个机场找气象站
def find_weather_station(lat, lon):
    stations = Stations()
    stations = stations.nearby(lat, lon)
    station = stations.fetch(1)
    return station.index[0] if not station.empty else None


airports_df['station_id'] = airports_df.apply(
    lambda row: find_weather_station(row['latitude'], row['longitude']),
    axis=1
)

print(f"映射完成，{airports_df['station_id'].notna().sum()}/{len(airports_df)} 个机场找到气象站")

# 3. 批量下载天气数据
all_weather = []

for idx, row in airports_df.iterrows():
    if pd.isna(row['station_id']):
        continue

    print(f"\n处理 {row['airport_code']} ({idx + 1}/{len(airports_df)})...")

    for year in [2018, 2019, 2020, 2021, 2022]:
        try:
            start = datetime(year, 1, 1)
            end = datetime(year, 12, 31, 23)

            data = Hourly(row['station_id'], start, end)
            df = data.fetch()

            if not df.empty:
                df['airport_code'] = row['airport_code']
                df.reset_index(inplace=True)
                all_weather.append(df)
                print(f"  ✓ {year}: {len(df)} 条记录")
            else:
                print(f"  ✗ {year}: 无数据")
        except Exception as e:
            print(f"  ✗ {year}: {e}")

        time.sleep(1)  # 避免请求过快

# 4. 合并并保存
if all_weather:
    combined = pd.concat(all_weather, ignore_index=True)

    # 保存为Parquet（PySpark友好且压缩率高）
    combined.to_parquet('weather_data_2018_2022.parquet', index=False)

    print(f"\n✓ 完成！总共 {len(combined)} 条记录")
    print(f"✓ 文件大小: {combined.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    print(f"✓ 保存为: weather_data_2018_2022.parquet")
else:
    print("\n✗ 未获取到任何数据")