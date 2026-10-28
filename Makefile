.PHONY: install install-all lint format typecheck test cov check download update-data fill-gaps clean eda prepare data train backtest evaluate verify dashboard all

install:          ## Dépendances de base + outils de dev + EDA
	uv sync --extra dev --extra eda

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

download:         ## CSV Kaggle quotidien + empreinte SHA-256 (KAGGLE_API_TOKEN ou ~/.kaggle/access_token)
	uv run gold-agents download

update-data:      ## Jours récents via Twelve Data (TWELVEDATA_API_KEY requis)
	uv run gold-agents update-data

fill-gaps:        ## Comble les trous du CSV Kaggle avec Twelve Data (TWELVEDATA_API_KEY requis)
	uv run gold-agents fill-gaps

clean:            ## Dataset propre dans data/processed/xau_1d_clean.parquet
	uv run gold-agents clean

eda:              ## Analyse exploratoire : rapport et figures dans outputs/eda/
	uv run gold-agents eda

prepare:          ## Features, cible et découpage dans data/processed/xau_1d_features.parquet
	uv run gold-agents prepare

data: clean prepare

train:
	uv run gold-agents train

backtest:         ## Buy & Hold + SMA 20/50 sur le test : résultats, registres, schéma
	uv run gold-agents backtest

evaluate:
	uv run gold-agents evaluate

verify:           ## Vérifie les registres de outputs/ledger/
	uv run gold-agents verify

dashboard:
	uv run streamlit run dashboard/app.py

all: data train backtest evaluate verify
