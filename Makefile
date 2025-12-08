.PHONY: help install dev test typecheck lint format security quality check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev:  ## Install dev dependencies and setup pre-commit
	uv sync --group dev
	uv run pre-commit install

test:  ## Run tests with coverage
	uv run pytest

typecheck:  ## Run mypy type checking
	uv run mypy biostructbenchmark/

lint:  ## Run linting checks
	uv run ruff check biostructbenchmark/ tests/
	uv run ruff format --check biostructbenchmark/ tests/

format:  ## Format code and auto-fix
	uv run ruff format biostructbenchmark/ tests/
	uv run ruff check --fix biostructbenchmark/ tests/

security:  ## Run security scans (Bandit + pip-audit)
	uv run bandit -c pyproject.toml -r biostructbenchmark/
	uv run pip-audit --ignore-vuln PYSEC-2022-42969

quality:  ## Run quality checks (docstring coverage + dead code)
	uv run interrogate -c pyproject.toml biostructbenchmark/
	uv run vulture biostructbenchmark/

check:  ## Run all checks (lint, typecheck, quality, test, security)
	@make lint && make typecheck && make quality && make test && make security

clean:  ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ .coverage
