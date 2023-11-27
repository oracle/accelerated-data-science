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
publish: clean
	sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" -b singlehtml && \
			oci os object put --name beta/index.html --file build/index.html -bn oci-feature-store -ns ociodscdev --force --content-type "text/html"

launch:
	docker run -it --network host $(IMAGE_NAME):$(TAG)
