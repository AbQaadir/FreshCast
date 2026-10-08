"""Training and benchmark execution pipeline for FreshCast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import yaml

from freshcast.data.generator import FoodserviceDataGenerator
from freshcast.data.loader import DataLoader
from freshcast.features.pipeline import TimeSeriesFeaturePipeline
from freshcast.metrics.financial import calculate_supply_chain_financial_loss
from freshcast.metrics.standard import calculate_standard_metrics
from freshcast.models.backtest import WalkForwardBacktester
from freshcast.models.baselines import MovingAverageForecaster, NaiveSeasonalForecaster
from freshcast.models.gbdt import LightGBMForecaster, XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("Initializing FreshCast Training & Benchmarking Pipeline...")

    # 1. Load or Generate Data
    data_path = Path("data/raw/foodservice_daily_sales.csv")
    if not data_path.exists():
        logger.info("Dataset not found. Running FoodserviceDataGenerator...")
        generator = FoodserviceDataGenerator()
        raw_df = generator.generate()
        generator.save(raw_df)

    loader = DataLoader()
    df = loader.load_raw_data(data_path)

    # 2. Configure Feature Pipeline
    pipeline = TimeSeriesFeaturePipeline(
        lags=[1, 2, 3, 7, 14, 21, 28],
        rolling_windows=[7, 14, 28],
        country="US",
    )

    # 3. Model Benchmark Hierarchy
    models = [
        NaiveSeasonalForecaster(lag=7),
        MovingAverageForecaster(window=7),
        LightGBMForecaster(),
        XGBoostForecaster(),
    ]

    # 4. Strict Walk-Forward Backtesting (3 Folds, 28-day window each)
    backtester = WalkForwardBacktester(n_splits=3, test_window_days=28)
    fold_results, summary = backtester.benchmark_models(models, df, pipeline)

    # Save benchmark results
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(processed_dir / "backtest_results.csv", index=False)
    summary.to_csv(processed_dir / "backtest_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("WALK-FORWARD BENCHMARK SUMMARY (Mean across 3 sequential out-of-time folds)")
    print("=" * 80)
    display_summary = summary.copy()
    display_summary["wape"] = (display_summary["wape"] * 100).round(2).astype(str) + "%"
    display_summary["mae"] = display_summary["mae"].round(2)
    display_summary["rmse"] = display_summary["rmse"].round(2)
    display_summary["financial_loss_usd"] = (
        "$" + display_summary["total_financial_loss_usd"].round(2).map("{:,.2f}".format)
    )
    display_summary["spoilage_usd"] = (
        "$" + display_summary["spoilage_loss_usd"].round(2).map("{:,.2f}".format)
    )
    display_summary["stockout_usd"] = (
        "$" + display_summary["stockout_loss_usd"].round(2).map("{:,.2f}".format)
    )
    print(
        display_summary[
            ["model", "wape", "mae", "rmse", "financial_loss_usd", "spoilage_usd", "stockout_usd"]
        ].to_string(index=False)
    )
    print("=" * 80 + "\n")

    # 5. Train Champion Model on Full Historical Train Split & Evaluate Out-of-Time
    train_df, test_df = loader.split_train_test_by_date(df, test_days=28)
    champion_pipeline = TimeSeriesFeaturePipeline(
        lags=[1, 2, 3, 7, 14, 21, 28],
        rolling_windows=[7, 14, 28],
    )
    train_transformed = champion_pipeline.fit_transform(train_df, drop_na=True)
    X_train = train_transformed[champion_pipeline.feature_cols]
    y_train = train_transformed[champion_pipeline.TARGET_COL]

    champion_model = LightGBMForecaster()
    champion_model.fit(X_train, y_train)

    # Out-of-time test evaluation
    combined_test = pd.concat([train_df, test_df], ignore_index=True)
    combined_transformed = champion_pipeline.transform(combined_test, drop_na=False)
    test_transformed = combined_transformed[
        combined_transformed["date"] >= test_df["date"].min()
    ].copy()

    X_test = test_transformed[champion_pipeline.feature_cols]
    y_test = test_transformed[champion_pipeline.TARGET_COL]
    y_pred = champion_model.predict(X_test)
    test_transformed["pred_sales"] = y_pred

    test_metrics = calculate_standard_metrics(y_test.values, y_pred)
    test_financial = calculate_supply_chain_financial_loss(
        test_transformed, pred_col="pred_sales", actual_col=champion_pipeline.TARGET_COL
    )

    logger.info(
        "Final Out-Of-Time Holdout Performance: WAPE=%.2f%%, MAE=%.2f, Financial Loss=$%.2f",
        test_metrics["wape"] * 100,
        test_metrics["mae"],
        test_financial["total_financial_loss_usd"],
    )

    feature_imp = champion_model.get_feature_importance()
    feature_imp.to_csv(processed_dir / "feature_importance.csv", index=False)

    # 6. Serialize Champion Bundle
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": champion_model,
        "pipeline": champion_pipeline,
        "feature_importance": feature_imp,
        "test_metrics": test_metrics,
        "test_financial": test_financial,
        "evaluation_df": test_transformed,
    }
    model_path = models_dir / "champion_model.joblib"
    joblib.dump(bundle, model_path)
    logger.info("Successfully serialized champion model bundle to %s", model_path)

    return bundle


if __name__ == "__main__":
    run_pipeline()
