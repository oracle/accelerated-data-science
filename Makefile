UV_CACHE_DIR ?= .uv-cache

dist: clean
	@UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --no-project --with build python -m build --sdist --wheel --outdir dist

check-dist: dist
	@UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --no-project --with twine python -m twine check dist/*
	@UV_CACHE_DIR=$(UV_CACHE_DIR) uv pip install --system dist/*.whl
	@python -c "import ads; print(ads.__version__)"

publish: check-dist
	@UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --no-project --with twine python -m twine upload dist/*

clean:
	@echo "Cleaning - removing dist, *.pyc, Thumbs.db and other files"
	@rm -rf dist build oracle_ads.egg-info
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@find ./ -name '.DS_Store' -exec rm -f {} \;
