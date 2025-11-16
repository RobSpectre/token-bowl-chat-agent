.PHONY: help
help: ## Show this help message
	@echo "Token Bowl Chat Agent - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package in development mode
	pip install -e ".[dev,mcp]"

.PHONY: test
test: ## Run tests
	pytest tests/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage
	pytest tests/ --cov=src/token_bowl_chat_agent --cov-report=term-missing

.PHONY: format
format: ## Format code with ruff
	ruff format src/ tests/

.PHONY: lint
lint: ## Lint code with ruff
	ruff check src/ tests/

.PHONY: type-check
type-check: ## Type check with mypy
	mypy src/

.PHONY: ci
ci: lint type-check test ## Run all CI checks

.PHONY: build
build: ## Build distribution packages
	python -m build

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.PHONY: publish-test
publish-test: clean build ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

.PHONY: publish
publish: clean build ## Publish to PyPI
	python -m twine upload dist/*