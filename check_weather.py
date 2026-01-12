import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "flights_with_weather_sample.parquet")

df = pd.read_parquet(SAMPLE_PATH)
weather_cols = ["temp", "dwpt", "rhum", "prcp", "snow",
                "wdir", "wspd", "wpgt", "pres", "tsun", "coco"]

print("Sample shape:", df.shape)
for col in weather_cols:
    if col in df.columns:
        missing_ratio = df[col].isna().mean()
        print(f"{col}: missing = {missing_ratio:.3f}")
