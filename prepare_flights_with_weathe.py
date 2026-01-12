import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLIGHTS_PATH = os.path.join(BASE_DIR, "flights_2018_2022_clean.parquet")
WEATHER_PATH = os.path.join(BASE_DIR, "weather_data_filled.parquet")

OUTPUT_PATH = os.path.join(BASE_DIR, "flights_with_weather_sample.parquet")


def main():
    print("[INFO] Loading data...")
    flights = pd.read_parquet(FLIGHTS_PATH)
    weather = pd.read_parquet(WEATHER_PATH)

    print("[INFO] Flights:", flights.shape)
    print("[INFO] Weather:", weather.shape)

    # ============ 1. 处理 flights 的 dep_hour ============
    print("[INFO] Extracting dep_hour from CRSDepTime...")

    flights["CRSDepTime"] = pd.to_numeric(flights["CRSDepTime"], errors="coerce")
    flights["dep_hour"] = (flights["CRSDepTime"] // 100).astype("Int64")

    # 确保 FlightDate 是字符串格式
    flights["FlightDate"] = flights["FlightDate"].astype(str)

    # ============ 2. 将 weather 的 time 字段拆成 date + hour ============
    print("[INFO] Parsing weather time into date & hour...")

    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    weather["date"] = weather["time"].dt.date.astype(str)
    weather["hour"] = weather["time"].dt.hour.astype("Int64")

    # 机场代码统一命名为 Origin（用于 join）
    weather.rename(columns={"airport_code": "Origin"}, inplace=True)

    # ============ 3. 只保留天气所需的列 ============
    weather_needed_cols = ["Origin", "date", "hour",
                           "temp", "dwpt", "rhum", "prcp", "snow",
                           "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]

    weather_small = weather[weather_needed_cols]

    print("[INFO] Clean weather shape:", weather_small.shape)

    # ============ 4. 从 flights 中取一个样本 join（20万行即可） ============
    flights_sample = flights.sample(min(200000, len(flights)), random_state=42)

    print("[INFO] Joining on keys: (Origin, FlightDate, dep_hour)")

    merged = flights_sample.merge(
        weather_small,
        left_on=["Origin", "FlightDate", "dep_hour"],
        right_on=["Origin", "date", "hour"],
        how="left",
    )

    print("[INFO] Merge finished. Shape:", merged.shape)
    print(merged.head())

    print(f"[INFO] Saving sample merged data to: {OUTPUT_PATH}")
    merged.to_parquet(OUTPUT_PATH, index=False)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
