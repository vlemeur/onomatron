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

# Run all Git hooks
check:
    uv run --locked prek run --all-files

# Run every quality check
all:
    just lint
    just type-check
    just check
