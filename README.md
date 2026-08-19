# litellm-proxy-demo

Two demos that talk to a [LiteLLM](https://github.com/BerriAI/litellm) proxy serving local models over an OpenAI-compatible API: the **LiteLLM SDK demo** (call the proxy from Python) and the **pi coding agent** ([pi](https://pi.dev/docs/latest/), a coding harness, pointed at the same proxy). The models are hosted by the proxy operator, not on your machine: you call them remotely with a base URL and an API key. Everything client-side runs inside a Docker sandbox.

## Setup

Requires [just](https://github.com/casey/just) and a container runtime (Docker Desktop, [Rancher Desktop](https://rancherdesktop.io), or [Colima](https://colima.io)).

```bash
just setup   # copies .env.example to .env and opens it for editing
```

The recipes live in [justfile](justfile). See [.env.example](.env.example) for what goes in `.env` (the proxy API key plus pi privacy settings).

## The sandbox (Docker)

Everything runs in one sandboxed container with this repo mounted at `/workspace`. Get in with `just shell`:

```sh
pi --list-models
pi
pi -p "hello" --model litellm/local/qwen3.8-27b-5090   # non-interactive
```

Inside a pi session, pick a model with `/model` (the catalog comes from [pi/models.json](pi/models.json)).

## LiteLLM SDK demo

[scripts/demo.py](scripts/demo.py) and [notebooks/demo.ipynb](notebooks/demo.ipynb) are the same demo in two formats: list the proxy's models, then send a hello-world completion and print the reply with token usage.

```bash
just demo      # script
just notebook  # Jupyter on http://localhost:8888 (token-less, localhost only)
```

## Security

Inference happens on the proxy host, but everything else runs in the sandbox container on your machine and can execute code (including the notebook kernels).

<details>
<summary>What the container can and can't do</summary>

The container needs internet access to call the proxy. It can also reach any other site, and the agent can read the repo and run code — a prompt it processes could tell it to send data somewhere. Limits:

- only the repo is mounted, at `/workspace`; nothing else from the host is reachable
- the filesystem is read-only except `/tmp`, which is wiped when the container stops; installed packages disappear with it
- all Linux capabilities are dropped and `no-new-privileges` is set, so nothing inside can escalate
- if the agent, its packages, or the kernel has a container-escape bug, the container lands on the host with the Docker daemon's rights (often root). The limits above shrink that surface; they don't remove it. Run this on a machine or VM you can spare.

</details>

- Keep the repo clean. Don't put private data, keys, or credentials in this repo while anything runs in the sandbox. The proxy key in `.env` is the only secret it can see, and `.env` is gitignored.
- Run the demos only through the sandbox recipes. If you ever run the code outside Docker, use a throwaway VM.
