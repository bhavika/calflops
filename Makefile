SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)

.PHONY:	style test

# Run code quality checks
style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix

build_dist:
	rm -fr build
	rm -fr dist
	python -m build
