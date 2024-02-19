# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

IMAGE_NAME:=oci-featurestore-docs
TAG:=preview
REGISTRY:=iad.ocir.io/ociodscdev
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is me`ant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*

build-image: clean html
	docker build --network host -t $(IMAGE_NAME):$(TAG) .

push: build-image
	docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(IMAGE_NAME):$(TAG) && \
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

# https://objectstorage.us-ashburn-1.oraclecloud.com/p/vv1bNIvcMnbvbM5TUW5NDVSoisL4BAS54l9bGFDmyiNkTPZ4q6NtAmOdemNr8lTv/n/ociodscdev/b/oci-feature-store/o/beta/index.html#document-quickstart
publish: html
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --prefix beta/ --overwrite
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --include "*.js" --content-type "text/javascript" --prefix beta/ --overwrite
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --include "*.css" --content-type "text/css" --prefix beta/ --overwrite
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --include "*.gif" --content-type "image/gif" --prefix beta/ --overwrite
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --include "*.png" --content-type "image/png" --prefix beta/ --overwrite
	oci os object bulk-upload -bn oci-feature-store --src-dir build -ns ociodscdev --include "*.html" --content-type "text/html" --prefix beta/ --overwrite


launch:
	docker run -it --network host $(IMAGE_NAME):$(TAG)
