files = trading_state test *.py
test_files = *
# test_files = state
# test_files = state_errors
# test_files = state*
# test_files = allocate
# test_files = symbol

# For local development
test:
	PYTHONTRACEMALLOC=20 pytest -s -v test/test_$(test_files).py --doctest-modules --cov trading_state --cov-config=.coveragerc --cov-report term-missing

# For github actions
test-ci:
	pytest -s -v test/test_$(test_files).py --doctest-modules --cov trading_state --cov-config=.coveragerc --cov-report=xml

lint:
	@echo "Running ruff..."
	@ruff check $(files)
	@echo "Running mypy..."
	@mypy $(files)

fix:
	ruff check --fix $(files)

install:
	pip install -U .[dev]

install-all:
	pip install -U .[dev,doc]

report:
	codecov

build: trading_state
	rm -rf dist
	python -m build

publish:
	make build
	twine upload --config-file ~/.pypirc -r pypi dist/*

.PHONY: test build
