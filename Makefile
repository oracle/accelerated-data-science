dist: clean
	@python3 -m build

publish: dist
	@twine upload dist/*

clean:
	@echo "Cleaning - removing dist, *.pyc, Thumbs.db and other files"
	@rm -rf dist build oracle_ads.egg-info
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@find ./ -name '.DS_Store' -exec rm -f {} \;

aqua.test:
	pip install -e .
	jupyter server extension enable --py ads.aqua.extension
	jupyter lab --NotebookApp.disable_check_xsrf=True --no-browser