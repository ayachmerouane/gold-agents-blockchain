.PHONY: install install-all lint format typecheck test cov check download update-data data train backtest evaluate verify dashboard all

install:          ## Dépendances de base + outils de dev
	uv sync --extra dev

install-all:      ## Tout, y compris PyTorch et le tableau de bord
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow"

cov:
	uv run pytest -m "not slow" --cov --cov-report=term-missing

check: lint typecheck test

download:         ## CSV Kaggle quotidien + empreinte SHA-256 (jeton Kaggle requis)
	uv run gold-agents download

update-data:      ## Jours récents via Twelve Data (TWELVEDATA_API_KEY requis)
	uv run gold-agents update-data

data:
	uv run gold-agents prepare

train:
	uv run gold-agents train

backtest:
	uv run gold-agents backtest

evaluate:
	uv run gold-agents evaluate

verify:
	uv run python scripts/verify_ledger.py

dashboard:
	uv run streamlit run dashboard/app.py

all: data train backtest evaluate verify
