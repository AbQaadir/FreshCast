.PHONY: setup generate-data train evaluate test dashboard clean

PYTHON := uv run python
STREAMLIT := uv run streamlit
PYTEST := uv run pytest

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

generate-data:
	$(PYTHON) -m freshcast.data.generator

train:
	$(PYTHON) -m freshcast.models.train

evaluate:
	$(PYTHON) -m freshcast.metrics.evaluate

test:
	$(PYTEST) tests/ -v

dashboard:
	$(STREAMLIT) run app/streamlit_app.py

clean:
	rm -rf .pytest_cache data/processed models/*.joblib
