# Task runner for this repo.
#
# Install just (a make-like runner) if you don't have it:
#   macOS:      brew install just
#   Linux:      https://github.com/casey/just#installation
#   or:         cargo install just
#
# List recipes: just --list

# --- one-time setup ---

# Create .env from the template, then prompt you to fill in the API key
setup:
    cp -n .env.example .env
    @echo "Edit .env and set LITELLM_API_KEY"
    $EDITOR .env

# --- sandbox (Docker) ---
# All commands run inside the sandbox container; nothing runs on the host.
# --rm deletes the container when it exits, so no containers pile up.

# Drop into a shell in the sandbox (run pi yourself once inside)
shell:
    docker compose run --rm pi

# Script demo: list proxy models, send a hello-world completion.
demo:
    docker compose run --rm pi -c "uv sync --frozen && .venv/bin/python scripts/demo.py"

# Jupyter notebook demo, published on http://localhost:8888.
notebook:
    docker compose run --rm --init -p 8888:8888 pi -c "uv sync --frozen && .venv/bin/jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --ServerApp.token= --ServerApp.root_dir=/workspace"
