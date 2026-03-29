# Documentation

## Getting Started

Install `uv` using the instructions at https://docs.astral.sh/uv/getting-started/installation/.
Then create a virtual environment for docs work:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Install relevant development packages.

```bash
uv pip install -r requirements.txt
```

## Developing Docs With Live Reload

Start live-reload during the development face

```bash
uv run sphinx-autobuild source/ build/
```

Open in the browser [http://127.0.0.1:8000]

## Build

To build and create the html documentation, run the following in the `docs/` folder.

```bash
uv run sphinx-build -b html source/ docs_html/
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
