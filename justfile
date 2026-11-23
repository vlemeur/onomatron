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

# Test suite with a coverage report (terminal + htmlcov/ + coverage.xml)
test-cov:
    uv run --locked pytest --cov --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml

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

# Export the best model to ONNX for the web demo (writes demo/model.onnx + demo/vocab.json);
# needs the "export" extra. Extra args are forwarded, e.g.:
#   just export-onnx --model wavenet --out demo/
export-onnx *args:
    uv run --locked --extra export python scripts/export_onnx.py {{args}}

# Serve the web demo locally (http://localhost:8000)
demo:
    uv run --locked python -m http.server 8000 --directory demo
