SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)

.PHONY:	style test

style:
	black .
	ruff check calflops examples --fix

build_dist:
	rm -fr build
	rm -fr dist
	python -m build
