# litellm-proxy-demo

A playground for talking to a [LiteLLM](https://github.com/BerriAI/litellm) proxy that serves local models over an OpenAI-compatible API. The models are hosted by the proxy operator, not on your machine: you call them remotely with a base URL and an API key. Everything client-side runs inside a Docker sandbox: the demos, the notebooks, and the agent.

Two demos:

1. **LiteLLM SDK demo** (Python). Call the proxy directly from code. [scripts/demo.py](scripts/demo.py) and [notebooks/demo.ipynb](notebooks/demo.ipynb) are the same demo in two formats.
2. **pi coding agent**. Run [pi](https://pi.dev/docs/latest/) (a coding harness) in the sandbox, pointed at the same proxy.

## Setup

Requirements:

- [just](https://github.com/casey/just) (a make-like task runner). On macOS: `brew install just`.
- A container runtime: Docker Desktop, [Rancher Desktop](https://rancherdesktop.io), or [Colima](https://colima.io).
- `uv` on the host, only if you want to re-resolve the lockfile (see below).

```bash
just setup   # copies .env.example to .env and opens it for editing
```

The recipes live in [justfile](justfile); each works as a plain shell command if you skip just. See [.env.example](.env.example) for what goes in `.env` (the proxy API key plus pi privacy settings).

### Python dependencies

[uv.lock](uv.lock) pins every dependency, including transitive ones. The container installs exactly those versions on each demo run.

Re-resolving the lockfile is a manual step: run `uv lock --upgrade` on your schedule (e.g. weekly), review the diff, and commit it. Resolution is limited by a 7-day cooldown in [pyproject.toml](pyproject.toml) so a bad PyPI release has to sit in public for a week before it can be picked up.

## The sandbox (Docker)

All demos and the agent run in the same sandboxed container with this repo mounted at `/workspace`. The container's entrypoint is a shell: get in with `just shell`, then run whatever you want yourself.

```bash
just shell
```

Inside the shell, run the pi coding agent, pointed at the proxy:

```sh
pi --list-models
pi
pi -p "hello" --model litellm/local/qwen3.8-27b-5090   # non-interactive
```

Inside a pi session, pick a model with `/model` (the catalog comes from [pi/models.json](pi/models.json)).

<details>
<summary>How the sandbox is locked down</summary>

The container can reach the internet. It needs that to call the LLM proxy, but it could also reach any other site. The agent reads everything in `/workspace` and can run code, so if a prompt it processes tells it to send data somewhere, it can. What the sandbox blocks:

- **host files**: the repo is the only thing mounted in, at `/workspace`
- **writes**: the container filesystem is read-only except `/tmp`, which is a temporary filesystem that is wiped when the container stops
- **privilege escalation**: all Linux capabilities are dropped and the container cannot gain new ones (`no-new-privileges`)
- **container escape**: if a bug in the agent, its packages, or the kernel lets it break out of the container, it lands on the host with whatever rights the Docker daemon has (often root on the daemon's filesystem). The hardening above shrinks the surface for this; it can't remove it. Run it on a machine or VM you can spare.

What it allows:

- internet access
- installing packages (the image ships python3, uv, and npm; see the Dockerfile), which land in the read-only fs's writable layers and disappear with the container
- pi telemetry and startup version check are off (`PI_TELEMETRY` in `.env`)

</details>

`--rm` deletes the container when it exits, so finished sessions don't pile up as stopped containers. You don't need a `docker compose build` step first: compose builds the image automatically when it isn't present yet.

## LiteLLM SDK demo

Runs in the sandbox. Lists the models on the proxy, then sends a hello-world completion to the default model and prints the reply with token usage.

```bash
just demo
```

### Notebook version

[notebooks/demo.ipynb](notebooks/demo.ipynb) is the same demo as a Jupyter notebook, also run in the sandbox: jupyterlab is published on port 8888.

```bash
just notebook
```

Then open `notebooks/demo.ipynb` in the browser at the URL Jupyter prints (typically `http://localhost:8888/notebooks/demo.ipynb`). The server is token-less; it binds to `localhost` only.

## Security

Inference happens on the proxy host, but everything else runs in the sandbox container on your machine and can execute code (including the notebook kernels).

- Keep the repo clean. Don't put private data, keys, or credentials in this repo while anything runs in the sandbox. The proxy key in `.env` is the only secret it can see, and `.env` is gitignored.
- Use the sandbox recipes. They are the only way to run the demos; the container's hardening is documented in [the lockdown note above](#the-sandbox-docker). If you ever run the code outside Docker, use a throwaway VM.
- Don't leave an agent running unattended. The SDK demo is a plain script; the coding agent is the part that acts.
- The notebook server runs without a token but is published only to `localhost`; don't forward port 8888 to other hosts.
- Telemetry is off for pi (`PI_TELEMETRY=0`), and the LiteLLM SDK sends requests only to your proxy.
