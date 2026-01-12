import pandas as pd

# 本地路径：你已经确认路径无误
path = "/Users/jorahmormont/PycharmProjects/BigDataFinalProject/flights_with_weather_propagation_by_year/flights_weather_propagation_2018.parquet"

# 配置
TARGET_COL = "DepDel15"
LEAKAGE_KEYWORDS = [
    "DepDelay", "DepDel15", "ArrDelay", "ArrDel15", "DepTime", "ArrTime",
    "Cancelled", "Canceled", "Diverted", "Delay"
]
TIME_TRAVEL_KEYWORDS = ["arrtime", "arrival", "arrdelay", "destweather", "dest_wx"]

# 加载数据
df = pd.read_parquet(path)

# 检查 1：潜在信息泄露字段
leakage_cols = [col for col in df.columns if any(kw.lower() in col.lower() for kw in LEAKAGE_KEYWORDS)]
print("\n🚨 潜在数据泄露字段:")
for col in leakage_cols:
    print(f"  - {col}")
if not leakage_cols:
    print("  ✅ 无明显泄露字段")

# 检查 2：目标分布
print("\n📊 目标变量分布:")
if TARGET_COL in df.columns:
    counts = df[TARGET_COL].value_counts(dropna=False)
    total = counts.sum()
    for val, cnt in counts.items():
        pct = cnt / total * 100
        print(f"  - {val}: {cnt} ({pct:.2f}%)")
    if 0 in counts and 1 in counts:
        imbalance_ratio = min(counts[0], counts[1]) / max(counts[0], counts[1])
        if imbalance_ratio < 0.1:
            print("  ⚠️ 类别严重不平衡")
else:
    print("  ❌ 未找到目标列")

# 检查 3：高相关性数值特征
print("\n🔍 高相关性数值特征 (Pearson > 0.9):")
numeric_cols = df.select_dtypes(include="number").columns.drop(TARGET_COL, errors="ignore")
high_corr = []
for col in numeric_cols:
    try:
        corr = df[col].corr(df[TARGET_COL])
        if abs(corr) >= 0.9:
            high_corr.append((col, corr))
    except:
        continue
if high_corr:
    for col, corr in high_corr:
        print(f"  - {col}: {corr:.3f}")
else:
    print("  ✅ 无强相关特征")

# 检查 4：时间穿越风险
print("\n⏳ 时间穿越相关字段:")
risk_fields = [col for col in df.columns if any(kw in col.lower() for kw in TIME_TRAVEL_KEYWORDS)]
for col in risk_fields:
    print(f"  - {col}")
if not risk_fields:
    print("  ✅ 无明显时间穿越字段")
