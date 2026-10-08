"""Unit tests for FreshCast FastAPI endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from freshcast.api.app import app
from freshcast.data.generator import FoodserviceDataGenerator
from freshcast.models.train import run_pipeline


@pytest.fixture(scope="module")
def client():
    if not Path("data/raw/foodservice_daily_sales.csv").exists():
        gen = FoodserviceDataGenerator()
        df = gen.generate()
        gen.save(df)
    if not Path("models/champion_model.joblib").exists():
        run_pipeline()

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is True
    assert data["active_dcs"] == 3
    assert data["active_skus"] == 10


def test_metadata_endpoint(client):
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "distribution_centers" in data
    assert "skus" in data
    assert len(data["distribution_centers"]) == 3
    assert len(data["skus"]) == 10


def test_forecast_endpoint(client):
    payload = {
        "dc_id": "DC_BOS",
        "sku_id": "SKU_BEEF_SIRLOIN",
        "horizon_days": 14,
        "discount_pct": 0.10,
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dc_id"] == "DC_BOS"
    assert data["sku_id"] == "SKU_BEEF_SIRLOIN"
    assert data["horizon_days"] == 14
    assert len(data["daily_forecasts"]) == 14
    assert data["total_projected_demand"] > 0
    first_day = data["daily_forecasts"][0]
    assert "date" in first_day
    assert first_day["forecasted_demand"] >= 0
    assert first_day["lower_bound_p10"] <= first_day["forecasted_demand"]
    assert first_day["upper_bound_p90"] >= first_day["forecasted_demand"]


def test_replenishment_endpoint(client):
    payload = {
        "dc_id": "DC_BOS",
        "sku_id": "SKU_BEEF_SIRLOIN",
        "service_level": 0.95,
        "current_on_hand_inventory": 150,
    }
    response = client.post("/api/v1/replenish", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["safety_stock"] > 0
    assert data["reorder_point"] > data["safety_stock"]
    assert "reorder_recommended" in data
    assert data["spoilage_risk_alert"] is True  # Beef sirloin has 6 days shelf life
