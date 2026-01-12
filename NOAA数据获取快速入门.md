# NOAA天气数据获取 - 快速入门指南

## 📋 目标
为你的2018-2022航班延误预测项目获取天气数据

## 🎯 方案概述
使用 **meteostat** Python库获取历史天气数据（最简单推荐的方法）

---

## 第一步：安装依赖 (5分钟)

```bash
# 安装必要的Python库
pip install meteostat pandas pyarrow requests

# 验证安装
python -c "import meteostat; print('meteostat安装成功')"
```

---

## 第二步：快速测试 (5分钟)

运行测试脚本验证一切正常：

```bash
python test_weather_download.py
```

**预期输出**：
- ✓ 找到JFK附近的气象站
- ✓ 下载2018年1月一周的数据
- ✓ 生成 `jfk_weather_sample.csv` 文件

如果看到 ✓ 测试成功，说明环境配置正确！

---

## 第三步：准备机场列表 (15分钟)

### 方法A：使用提供的机场列表
```bash
python create_airport_mapping.py
```
这会生成 `us_airports_coordinates.csv`，包含美国最繁忙的100个机场。

### 方法B：从你的航班数据提取
```python
import pandas as pd
from pyspark.sql import SparkSession

# 启动Spark
spark = SparkSession.builder.appName("ExtractAirports").getOrCreate()

# 读取你的航班数据
flight_df = spark.read.csv('your_flight_data.csv', header=True)

# 提取所有唯一机场
origins = flight_df.select('Origin').distinct()
dests = flight_df.select('Dest').distinct()

# 合并并去重
all_airports = origins.union(dests).distinct()
all_airports.write.csv('my_airports.csv', header=True)
```

---

## 第四步：批量下载天气数据 (2-4小时)

### 完整工作流脚本

```python
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
    
    print(f"\n处理 {row['airport_code']} ({idx+1}/{len(airports_df)})...")
    
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
    print(f"✓ 文件大小: {combined.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"✓ 保存为: weather_data_2018_2022.parquet")
else:
    print("\n✗ 未获取到任何数据")
```

### 保存并运行
```bash
# 将上面的代码保存为 download_all_weather.py
python download_all_weather.py
```

**预计时间**：
- 100个机场 × 5年 = 约2-4小时
- 可以中断后继续运行（已下载的数据不会重复）

---

## 第五步：数据质量检查

```python
import pandas as pd

# 读取数据
df = pd.read_parquet('weather_data_2018_2022.parquet')

print("数据概览：")
print(f"总记录数: {len(df):,}")
print(f"时间跨度: {df['time'].min()} 到 {df['time'].max()}")
print(f"覆盖机场: {df['airport_code'].nunique()} 个")

print("\n字段列表：")
print(df.columns.tolist())

print("\n缺失值统计：")
print(df.isnull().sum())

print("\n各机场记录数：")
print(df['airport_code'].value_counts().head(10))
```

---

## 第六步：与航班数据合并 (PySpark)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_timestamp, broadcast

spark = SparkSession.builder \
    .appName("MergeWeatherData") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# 1. 读取航班数据
flight_df = spark.read.csv('your_flight_data.csv', header=True)

# 2. 读取天气数据
weather_df = spark.read.parquet('weather_data_2018_2022.parquet')

# 3. 准备时间字段
# 假设航班数据有 FlightDate 和 DepTime
flight_df = flight_df.withColumn(
    'flight_datetime',
    to_timestamp(concat(col('FlightDate'), col('DepTime')), 'yyyyMMddHHmm')
)

weather_df = weather_df.withColumn(
    'weather_hour',
    to_timestamp(col('time'))
)

# 4. 合并起飞机场天气（使用12小时前的天气作为"预测"特征）
flight_with_weather = flight_df.join(
    broadcast(weather_df.select(
        col('airport_code').alias('weather_airport'),
        col('weather_hour'),
        col('temp').alias('origin_temp_12h_ago'),
        col('wspd').alias('origin_wind_12h_ago'),
        col('prcp').alias('origin_precip_12h_ago'),
        # 添加其他需要的天气字段
    )),
    (col('Origin') == col('weather_airport')) &
    (col('weather_hour') == col('flight_datetime') - expr('INTERVAL 12 HOURS')),
    'left'
)

# 5. 类似地添加目的地机场天气
# ... (重复上面的逻辑，使用 Dest 字段)

# 6. 保存结果
flight_with_weather.write.parquet('flight_with_weather_features.parquet')

print("✓ 天气数据合并完成！")
```

---

## 🎯 重要的天气特征

对航班延误预测最重要的字段：

1. **temp** - 温度（°C）
   - 极端温度影响飞机性能
   
2. **wspd** - 风速（km/h）
   - 强风导致无法起降
   
3. **prcp** - 降水量（mm）
   - 雨雪影响能见度和跑道条件
   
4. **snow** - 降雪量（mm）
   - 需要除冰，导致严重延误
   
5. **wdir** - 风向（度）
   - 侧风影响起降

6. **pres** - 气压（hPa）
   - 低气压通常意味着恶劣天气

---

## 💡 特征工程建议

```python
import pandas as pd
import numpy as np

def create_weather_features(df):
    """创建有意义的天气特征"""
    
    # 1. 恶劣天气评分 (0-10)
    df['bad_weather_score'] = 0
    
    # 低能见度（假设你有这个字段）
    # df['bad_weather_score'] += np.where(df['visibility'] < 5, 3, 0)
    
    # 强风
    df['bad_weather_score'] += np.where(df['wspd'] > 40, 3, 
                                 np.where(df['wspd'] > 25, 2, 0))
    
    # 降水
    df['bad_weather_score'] += np.where(df['prcp'] > 10, 3,
                                 np.where(df['prcp'] > 2, 2, 0))
    
    # 降雪
    df['bad_weather_score'] += np.where(df['snow'] > 5, 4,
                                 np.where(df['snow'] > 0, 2, 0))
    
    # 极端温度
    df['bad_weather_score'] += np.where((df['temp'] < -10) | (df['temp'] > 38), 1, 0)
    
    # 2. 天气变化率
    df = df.sort_values(['airport_code', 'time'])
    df['temp_change_3h'] = df.groupby('airport_code')['temp'].diff(3)
    df['wind_change_3h'] = df.groupby('airport_code')['wspd'].diff(3)
    
    # 3. 是否有降水
    df['has_precipitation'] = (df['prcp'] > 0).astype(int)
    df['has_snow'] = (df['snow'] > 0).astype(int)
    
    # 4. 风寒指数（冬季）
    df['wind_chill'] = 13.12 + 0.6215*df['temp'] - \
                       11.37*(df['wspd']**0.16) + \
                       0.3965*df['temp']*(df['wspd']**0.16)
    
    return df

# 使用
weather_df = pd.read_parquet('weather_data_2018_2022.parquet')
weather_with_features = create_weather_features(weather_df)
weather_with_features.to_parquet('weather_with_features.parquet')
```

---

## 🚨 常见问题解决

### 问题1：某些机场找不到气象站
**解决**：使用附近大型机场的天气数据
```python
# 为小机场指定使用邻近大机场的气象站
small_airport_mapping = {
    'ISP': 'JFK',  # Long Island使用JFK的天气
    'HPN': 'LGA',  # White Plains使用LaGuardia的天气
    # ...
}
```

### 问题2：数据有缺失值
**解决**：使用插值或历史均值
```python
# 前后插值
df['temp'] = df.groupby('airport_code')['temp'].transform(
    lambda x: x.interpolate(method='linear')
)

# 或使用同期历史均值
df['temp'] = df.groupby(['airport_code', df['time'].dt.month, df['time'].dt.hour])['temp'].transform(
    lambda x: x.fillna(x.mean())
)
```

### 问题3：下载太慢
**解决**：使用多线程或直接下载NOAA ISD文件
```python
from concurrent.futures import ThreadPoolExecutor

def download_year(args):
    airport_code, station_id, year = args
    # ... 下载逻辑
    
with ThreadPoolExecutor(max_workers=5) as executor:
    tasks = [(code, station, year) 
             for code, station in zip(codes, stations)
             for year in range(2018, 2023)]
    executor.map(download_year, tasks)
```

---

## 📊 预期数据规模

- **100个机场 × 5年**
- 每个机场: ~43,800条记录 (365天 × 24小时 × 5年)
- 总记录数: ~4,380,000条
- 存储空间: 
  - CSV格式: ~2-3 GB
  - Parquet格式: ~500 MB - 1 GB（推荐）

---

## ✅ 检查清单

在开始你的项目之前，确保：

- [ ] meteostat库安装成功
- [ ] 测试脚本运行成功
- [ ] 机场坐标列表准备完成
- [ ] 了解完整工作流程
- [ ] 知道如何处理缺失数据
- [ ] 计划好特征工程策略
- [ ] 了解如何与PySpark集成

---

## 📞 需要帮助？

参考以下文件：
1. `noaa_weather_data_guide.py` - 详细技术文档
2. `test_weather_download.py` - 测试脚本
3. `create_airport_mapping.py` - 机场映射工具

祝项目顺利！🚀
