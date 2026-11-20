set default-list := true

# Synchronize runtime and development dependencies from the lockfile
sync:
    uv sync --locked

# Install the development environment and Git hooks
setup: sync
    uv run --locked prek install

# Check code style and formatting without changing files
lint:
    uv run --locked ruff check .
    uv run --locked ruff format --check .

# Apply code style and formatting fixes
format:
    uv run --locked ruff check . --fix
    uv run --locked ruff format .

# Check types
type-check:
    uv run --locked ty check

# Run the test suite (local, no network)
test:
    uv run --locked pytest

# Run all Git hooks
check:
    uv run --locked prek run --all-files

# Run every quality check
all:
    just lint
    just type-check
    just test
    just check

# Scrape character names from Wikipedia/Wikidata/HF into data/processed/names.txt (network)
scrape:
    uv run --locked python scripts/run_scraper.py

# Benchmark + hyperparameter tuning; extra args are forwarded, e.g.:
#   just bench                                (script defaults: --mode all --trials 30)
#   just bench --trials 50 --models mlp wavenet
bench *args:
    uv run --locked python scripts/benchmark.py {{args}}

# MLflow UI on the benchmark tracking DB (http://127.0.0.1:5000)
bench-ui:
    uv run --locked mlflow ui --backend-store-uri sqlite:///data/benchmark/mlflow.db

# Local Streamlit UI on the best published model (fetched from the Hub at launch)
run:
    uv run --locked streamlit run scripts/app_streamlit.py
