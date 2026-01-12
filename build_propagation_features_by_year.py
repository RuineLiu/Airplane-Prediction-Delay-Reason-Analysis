import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "flights_with_weather_by_year")
OUTPUT_DIR = os.path.join(BASE_DIR, "flights_with_weather_propagation_by_year")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [2018, 2019, 2020, 2021, 2022]


def hhmm_to_minutes(hhmm):
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    hours = (hhmm // 100).astype("Int64")
    minutes = (hhmm % 100).astype("Int64")
    return (hours * 60 + minutes).astype("float")


def compute_timestamp(df, time_col):
    df[time_col + "_minutes"] = hhmm_to_minutes(df[time_col])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Convert FlightDate to total minutes since epoch instead of datetime objects
    # First convert to numpy datetime64 in nanoseconds, then to minutes
    flight_date_ns = df["FlightDate"].astype('int64')  # nanoseconds since epoch
    flight_date_minutes = flight_date_ns / (1e9 * 60)  # convert to minutes since epoch

    return flight_date_minutes + df[time_col + "_minutes"]


def build_propagation_features(df):
    print("[INFO] Converting DepTime and ArrTime to timestamps...")
    df["DepTimestamp"] = compute_timestamp(df, "DepTime")
    df["ArrTimestamp"] = compute_timestamp(df, "ArrTime")

    df["ArrDelay"] = pd.to_numeric(df["ArrDelay"], errors="coerce").fillna(0)
    df["ArrDel15"] = pd.to_numeric(df["ArrDel15"], errors="coerce").fillna(0).astype(int)

    df["origin_delay_rate_past2h"] = np.nan
    df["origin_arr_delay_avg_past2h"] = np.nan
    df["origin_late_arrivals_past2h"] = np.nan

    window_minutes = 120

    arrivals_by_airport = {}
    print("[INFO] Indexing arrivals by airport...")

    for dest, group in df.groupby("Dest"):
        arr_times = group["ArrTimestamp"].to_numpy()
        arr_del15 = group["ArrDel15"].to_numpy()
        arr_delay = group["ArrDelay"].to_numpy()
        order = arr_times.argsort()
        arrivals_by_airport[dest] = {
            "arr_times": arr_times[order],
            "arr_del15": arr_del15[order],
            "arr_delay": arr_delay[order]
        }

    print("[INFO] Computing propagation features airport by airport...")

    for origin, group in df.groupby("Origin"):
        if origin not in arrivals_by_airport:
            df.loc[group.index, ["origin_delay_rate_past2h",
                                 "origin_arr_delay_avg_past2h",
                                 "origin_late_arrivals_past2h"]] = 0
            continue

        arr_times = arrivals_by_airport[origin]["arr_times"]
        arr_del15 = arrivals_by_airport[origin]["arr_del15"]
        arr_delay = arrivals_by_airport[origin]["arr_delay"]

        dep_times = group["DepTimestamp"].to_numpy()
        dep_index = group.index.to_numpy()
        order = dep_times.argsort()
        dep_times = dep_times[order]
        dep_index = dep_index[order]

        late_counts = np.zeros(len(dep_times), dtype=int)
        total_counts = np.zeros(len(dep_times), dtype=int)
        delay_sums = np.zeros(len(dep_times), dtype=float)

        j = 0
        for k, dep_t in enumerate(dep_times):
            start_t = dep_t - window_minutes
            while j < len(arr_times) and arr_times[j] < start_t:
                j += 1
            i = j
            while i < len(arr_times) and arr_times[i] < dep_t:
                i += 1
            n = i - j
            if n > 0:
                late_counts[k] = arr_del15[j:i].sum()
                delay_sums[k] = arr_delay[j:i].sum()
                total_counts[k] = n

        rate = np.divide(late_counts, total_counts, out=np.zeros_like(late_counts, dtype=float), where=total_counts > 0)
        avg_delay = np.divide(delay_sums, total_counts, out=np.zeros_like(delay_sums, dtype=float),
                              where=total_counts > 0)

        df.loc[dep_index, "origin_delay_rate_past2h"] = rate
        df.loc[dep_index, "origin_arr_delay_avg_past2h"] = avg_delay
        df.loc[dep_index, "origin_late_arrivals_past2h"] = late_counts

    return df


def main():
    for year in YEARS:
        print(f"\n[INFO] Processing year {year}...")
        input_path = os.path.join(INPUT_DIR, f"flights_with_weather_{year}.parquet")
        output_path = os.path.join(OUTPUT_DIR, f"flights_weather_propagation_{year}.parquet")

        df = pd.read_parquet(input_path)
        df_with_features = build_propagation_features(df)
        df_with_features.to_parquet(output_path, index=False)
        print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    main()