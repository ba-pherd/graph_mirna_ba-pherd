# syntax=docker/dockerfile:1

FROM ghcr.io/ba-pherd/platform/task-base:main

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.in,target=/app/requirements.in \
    pip-compile requirements.in && pip-sync

COPY . /app

CMD ["python", "graph_mirna/main_test.py"]