format: ## apply black code formatter
	black pdcast

lint: ## check style with flake8
	flake8 pdcast tests

mypy: ## check type hints
	mypy pdcast tests

isort: ## sort imports
	isort .

cqa: format isort lint mypy ## run all cqa tools

test-all: ## run tests on every Python version with tox
	tox --skip-missing-interpreters

coverage: ## check code coverage quickly with the default Python
	coverage run --source pdcast -m pytest
	coverage report -m
	coverage html
	# $(BROWSER) htmlcov/index.html
