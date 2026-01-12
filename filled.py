import pandas as pd
import numpy as np

# 定义邻近机场映射表
# 格式：{缺失数据的机场: 替代机场}
NEARBY_AIRPORT_MAPPING = {
    # 纽约地区
    'LGA': 'JFK',  # LaGuardia使用JFK数据（相距约15公里）
    'HPN': 'LGA',  # White Plains使用LaGuardia
    'ISP': 'JFK',  # Long Island使用JFK

    # 芝加哥地区
    'MDW': 'ORD',  # Midway使用O'Hare

    # 华盛顿地区
    'DCA': 'IAD',  # Reagan使用Dulles
    'BWI': 'DCA',  # Baltimore使用Reagan（如果DCA也缺失）

    # 休斯顿地区
    'HOU': 'IAH',  # Hobby使用Intercontinental

    # 达拉斯地区
    'DAL': 'DFW',  # Love Field使用DFW

    # 洛杉矶地区
    'BUR': 'LAX',  # Burbank使用LAX
    'ONT': 'LAX',  # Ontario使用LAX
    'SNA': 'LAX',  # Orange County使用LAX
    'LGB': 'LAX',  # Long Beach使用LAX

    # 旧金山湾区
    'OAK': 'SFO',  # Oakland使用SFO
    'SJC': 'SFO',  # San Jose使用SFO

    # 其他
    'PBI': 'FLL',  # Palm Beach使用Fort Lauderdale
    'SRQ': 'TPA',  # Sarasota使用Tampa
}


def fill_missing_with_nearby(weather_df, mapping=NEARBY_AIRPORT_MAPPING):
    """
    使用邻近机场的数据填补缺失机场的数据

    参数：
        weather_df: 天气数据DataFrame
        mapping: 机场映射字典

    返回：
        填补后的DataFrame
    """
    filled_data = []

    for missing_airport, source_airport in mapping.items():
        # 检查缺失机场的数据情况
        missing_data = weather_df[weather_df['airport_code'] == missing_airport]
        source_data = weather_df[weather_df['airport_code'] == source_airport]

        if len(source_data) > 0:
            # 复制源机场的数据
            filled_records = source_data.copy()
            filled_records['airport_code'] = missing_airport
            filled_records['data_source'] = f'from_{source_airport}'

            # 只使用缺失的时间段
            if len(missing_data) > 0:
                existing_times = set(missing_data['time'])
                filled_records = filled_records[~filled_records['time'].isin(existing_times)]

            filled_data.append(filled_records)
            print(f"✓ 为 {missing_airport} 从 {source_airport} 填补了 {len(filled_records)} 条记录")
        else:
            print(f"✗ {missing_airport}: 源机场 {source_airport} 也没有数据")

    # 合并原始数据和填补数据
    if filled_data:
        all_filled = pd.concat(filled_data, ignore_index=True)
        result = pd.concat([weather_df, all_filled], ignore_index=True)
        result = result.sort_values(['airport_code', 'time']).reset_index(drop=True)
        return result

    return weather_df


# 使用示例
weather_df = pd.read_parquet('weather_data_2018_2022.parquet')
print(f"填补前记录数: {len(weather_df)}")

weather_filled = fill_missing_with_nearby(weather_df)
print(f"填补后记录数: {len(weather_filled)}")

# 保存
weather_filled.to_parquet('weather_data_filled.parquet')