# Documentation

## Getting Started

Setup Conda.

Download the latest Miniforge installer for your machine from here: https://github.com/conda-forge/miniforge

For example for linux you can run the script below:

```bash
curl https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Create conda environment.

```bash
conda create -n ads-docs python=3.8
conda activate ads-docs
```

Install relevant development packages.

```bash
pip install -r requirements.txt
```

## Developing Docs With Live Reload

Start live-reload during the development face

```bash
sphinx-autobuild source/ build/
```

Open in the browser [http://127.0.0.1:8000]

## Build

To build and create the html documentation, run the following in the `docs/` folder.

```bash
sphinx-build -b html  source/ docs_html/
```

To `zip` the content of the html docs

```bash
zip -r ads-latest.zip docs_html/.
```

## Notes

- the `source/conf.py` defines most everything for the docs
- the `ads.rst` was auto-generated but then hand edited to remove tests package

## Contribute

Now you can make updates to the docs and contributed via PRs.
