RELEASE_BRANCH := release/ads
DOCS_RELEASE_BRANCH := release
CLONE_DIR := /tmp/advanced-ds
DOCS_CLONE_DIR := /tmp/ads-docs
COPY_INVENTORY := setup.py CONTRIBUTING.md LICENSE.txt MANIFEST.in README-development.md README.md SECURITY.md THIRD_PARTY_LICENSES.txt

prepare-release-branch: clean
	@git checkout master
	@git clean -xdf
	@git pull
	git checkout -b release/$(RELEASE_VERSION)

prepare-ads:
	@echo "Started advanced-ds clone at $$(date)"
	@git clone ssh://git@bitbucket.oci.oraclecorp.com:7999/odsc/advanced-ds.git --branch $(RELEASE_BRANCH)  --depth 1 $(CLONE_DIR)
	@echo "Finished cloning at $$(date)" 
	cp -r $(CLONE_DIR)/ads .
	$(foreach var,$(COPY_INVENTORY),cp $(CLONE_DIR)/$(var) .;)

prepare-docs: 
	@echo "Started ads_docs clone at $$(date)"
	@git clone ssh://git@bitbucket.oci.oraclecorp.com:7999/odsc/ads_docs.git --branch $(DOCS_RELEASE_BRANCH)  --depth 1 $(DOCS_CLONE_DIR)
	@echo "Finished cloning at $$(date)" 
	cp -r $(DOCS_CLONE_DIR)/source docs/ && cp $(DOCS_CLONE_DIR)/requirements.txt docs

prepare: prepare-release-branch prepare-ads prepare-docs

push: clean
	@bash -c 'if [[ $$(git branch | grep \*) == "* release/$(RELEASE_VERSION)" ]];then echo "Version matching current branch"; else echo "Set proper value to RELEASE_VERSION";exit 1 ; fi'	
	@git add .
	@git commit -m "Release version: $(RELEASE_VERSION)"
	@git push --set-upstream origin release/$(RELEASE_VERSION)

dist: clean
	@python3 setup.py sdist bdist_wheel

publish: dist
	@twine upload dist/*

clean:
	@rm -rf dist build oracle_ads.egg-info
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf $(CLONE_DIR)
	@rm -rf $(DOCS_CLONE_DIR)
