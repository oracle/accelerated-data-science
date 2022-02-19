
dist: clean
	@python3 setup.py sdist bdist_wheel

publish: dist
	@twine upload dist/*

clean:
	@rm -rf dist build oracle_ads.egg-info
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
