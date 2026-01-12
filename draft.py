import os
import pandas as pd

# 相对路径：从项目根目录运行这个脚本
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "robikscube",
    "flight-delay-dataset-20182022",
    "versions",
    "4"
)

OUTPUT_PATH = os.path.join(BASE_DIR, "flights_2018_2022_clean.parquet")

YEARS = [2018, 2019, 2020, 2021, 2022]

# 我们主要关心的列（如果有缺失，后面会做容错）
IMPORTANT_COLUMNS = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    # 航司信息
    "Marketing_Airline_Network",
    "Operating_Airline",
    # 机场与航线
    "OriginAirportID",
    "Origin",
    "DestAirportID",
    "Dest",
    # 时间与延误
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDel15",
    # 取消/备降
    "Cancelled",
    "Diverted",
    # 原因分解
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    # 距离
    "Distance",
    "DistanceGroup",
]

def load_one_year(year: int) -> pd.DataFrame:
    """
    载入单年数据，只保留需要的列，并过滤 Cancelled/Diverted 航班。
    """
    parquet_path = os.path.join(DATA_DIR, f"Combined_Flights_{year}.parquet")
    csv_path = os.path.join(DATA_DIR, f"Combined_Flights_{year}.csv")

    if os.path.exists(parquet_path):
        print(f"[INFO] Loading {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"[INFO] Loading {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No Combined_Flights file found for year {year}")

    print(f"[INFO] Loaded {year} with shape {df.shape}")

    # 只保留存在的列（有些列可能在这个版本数据集中叫法略有不同）
    existing_cols = [c for c in IMPORTANT_COLUMNS if c in df.columns]
    missing_cols = [c for c in IMPORTANT_COLUMNS if c not in df.columns]

    if missing_cols:
        print(f"[WARN] Year {year}: missing columns: {missing_cols}")

    df = df[existing_cols].copy()

    # 过滤掉取消和备降的航班（如果有这些列）
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != 1]
    if "Diverted" in df.columns:
        df = df[df["Diverted"] != 1]

    print(f"[INFO] After filtering Cancelled/Diverted: {df.shape}")

    return df


def main():
    all_years = []

    for year in YEARS:
        df_year = load_one_year(year)
        all_years.append(df_year)

    print("[INFO] Concatenating all years...")
    df_all = pd.concat(all_years, ignore_index=True)
    print(f"[INFO] Final combined shape: {df_all.shape}")

    # 保存为 parquet，后续特征工程和 join 都用这个文件
    print(f"[INFO] Saving to {OUTPUT_PATH}")
    df_all.to_parquet(OUTPUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
