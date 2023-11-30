test:
	python -m unittest tests/test_*.py

style:
	black --line-length 80 aria

PHONY: test style