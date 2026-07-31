install:
	pip install -e ".[test,lint]"

test:
	pytest tests

format:
	ruff format src/ tests/ && ruff check --fix src/ tests/

lint:
	ruff check src/ tests/

all: format lint test
