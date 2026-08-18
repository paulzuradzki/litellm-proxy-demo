FROM node:22-alpine

RUN npm install -g @earendil-works/pi-coding-agent \
 && apk add --no-cache python3 git curl \
 && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"
