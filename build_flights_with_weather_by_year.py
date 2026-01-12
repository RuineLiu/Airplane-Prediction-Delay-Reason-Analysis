import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLIGHTS_PATH = os.path.join(BASE_DIR, "flights_2018_2022_clean.parquet")
WEATHER_PATH = os.path.join(BASE_DIR, "weather_data_filled.parquet")

OUTPUT_DIR = os.path.join(BASE_DIR, "flights_with_weather_by_year")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2019, 2020, 2021, 2022]


def main():
    print("[INFO] Loading base data...")
    flights = pd.read_parquet(FLIGHTS_PATH)
    weather = pd.read_parquet(WEATHER_PATH)

    # 处理 flights 的 dep_hour & FlightDate 格式
    flights["CRSDepTime"] = pd.to_numeric(flights["CRSDepTime"], errors="coerce")
    flights["dep_hour"] = (flights["CRSDepTime"] // 100).astype("Int64")
    flights["FlightDate"] = flights["FlightDate"].astype(str)

    # 处理 weather 的 date/hour 与机场代码
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    weather["date"] = weather["time"].dt.date.astype(str)
    weather["hour"] = weather["time"].dt.hour.astype("Int64")
    weather.rename(columns={"airport_code": "Origin"}, inplace=True)

    weather_needed_cols = ["Origin", "date", "hour",
                           "temp", "dwpt", "rhum", "prcp", "snow",
                           "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]
    weather_small = weather[weather_needed_cols]

    print("[INFO] Clean flights:", flights.shape)
    print("[INFO] Clean weather:", weather_small.shape)

    # 按年份循环
    for year in YEARS:
        print(f"\n[INFO] Processing year {year} ...")

        flights_y = flights[flights["Year"] == year].copy()
        print(f"[INFO] Flights {year} shape: {flights_y.shape}")

        weather_y = weather_small[weather["time"].dt.year == year].copy()
        print(f"[INFO] Weather {year} shape: {weather_y.shape}")

        print("[INFO] Joining on (Origin, FlightDate, dep_hour)...")
        merged_y = flights_y.merge(
            weather_y,
            left_on=["Origin", "FlightDate", "dep_hour"],
            right_on=["Origin", "date", "hour"],
            how="left",
        )

        print(f"[INFO] Merged {year} shape:", merged_y.shape)

        out_path = os.path.join(OUTPUT_DIR, f"flights_with_weather_{year}.parquet")
        print(f"[INFO] Saving {year} data to {out_path}")
        merged_y.to_parquet(out_path, index=False)

    print("\n[INFO] All years done.")


if __name__ == "__main__":
    main()
