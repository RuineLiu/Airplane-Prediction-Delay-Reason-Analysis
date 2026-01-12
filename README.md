# Strategic Flight Delay Prediction (Cost-Sensitive)

This project builds a strategic (T-12/T-24) flight delay prediction pipeline for
U.S. domestic flights, with an explicit cost-sensitive objective. Instead of
optimizing for raw accuracy on imbalanced data, the work prioritizes recall and
operational cost reduction, following the principle: better a false alarm than a
missed delay.

The accompanying article is in `Data Analysis/A Cost-Sensitive Approach .pdf`.

## Project Summary
- Scope: 2018-2022 U.S. BTS on-time performance data with Meteostat weather.
- Prediction horizon: 12-24 hours before departure (strategic planning).
- Core features: scheduled traffic congestion + historical network inertia.
- Cost-sensitive learning: false negatives cost 5x false positives.
- Key result (paper): ROC AUC ~0.70, recall ~68-70%, ~42% cost reduction vs
  zero-rule baseline, despite lower accuracy (~61%).

## Repository Layout
- `data_prepare.py`: download the flight dataset via kagglehub.
- `download_all_weather.py`: pull hourly weather from Meteostat.
- `filled.py`: fill missing airport weather using nearby airport mapping.
- `build_flights_with_weather_by_year.py`: join flights with weather by year.
- `build_propagation_features_by_year.py`: build propagation features by year.
- `prepare_flights_with_weathe.py`: small sample join for quick checks.
- `train_logistic.py`: logistic baseline with one-hot + scaling.
- `train_lgbm.py`: LightGBM template (update dataset path before running).
- `Data Analysis/Analyze flight delay parquet.py`: PySpark exploratory analysis.
- `Data Analysis/*.csv`: exported analysis tables and rankings.
- `NOAA数据获取快速入门.md`: Meteostat-based weather download guide (CN).

## Data Sources
- Flight data: BTS On-Time Performance (2018-2022).
- Weather data: Meteostat hourly observations (nearest airport stations).

Large datasets are intentionally excluded from the repo. The working files
include `.parquet` outputs such as:
- `flights_2018_2022_clean.parquet`
- `weather_data_2018_2022.parquet`
- `weather_data_filled.parquet`
- `flights_with_weather_by_year/*.parquet`
- `flights_with_weather_propagation_by_year/*.parquet`

## Methodology (from the paper)
1) Strategic horizon (T-12/T-24): avoid real-time features that are not
   available a day ahead.
2) Leakage-free features:
   - Scheduled traffic congestion: hourly scheduled dep/arr volumes.
   - Historical network inertia: rolling delay rates by origin, airline, route.
3) Cost-sensitive learning:
   - Total_Cost = 5 * FN + 1 * FP
   - Implemented via class weighting (e.g., LightGBM scale_pos_weight).

## Empirical Insights (from analysis + paper)
- Delay causes (total minutes): Late Aircraft (~41.5%), Carrier (~37.1%),
  NAS (~14.9%), Weather (~6.2%), Security (~0.2%).
- Temporal dynamics dominate: early morning flights are least delayed;
  delays accumulate through the day.
- Short-term history matters more than long-term averages (7-day > 30-day).
- Strategic models trade accuracy for recall but yield higher operational value.

## Workflow (Suggested)
1) Download flight data
   ```bash
   python data_prepare.py
   ```
2) Download weather data (see `NOAA数据获取快速入门.md`)
   ```bash
   python download_all_weather.py
   ```
3) Fill missing weather via nearby airports
   ```bash
   python filled.py
   ```
4) Join flights + weather and build features
   ```bash
   python build_flights_with_weather_by_year.py
   python build_propagation_features_by_year.py
   ```
5) Train models
   ```bash
   python train_logistic.py
   # Update path in train_lgbm.py before running
   python train_lgbm.py
   ```

## Paper Citation
R. Liu, "A Cost-Sensitive Approach to Strategic Flight Delay Prediction."
Lehigh University, 2018-2022 U.S. flight data (28M records) + Meteostat.

## Notes
- PySpark is used for large-scale analysis in `Data Analysis/`.
- The project emphasizes robust, cross-year evaluation using time-based splits:
  train (2018-2020), val (2021), test (2022).
