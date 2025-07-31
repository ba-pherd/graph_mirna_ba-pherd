# syntax=docker/dockerfile:1

FROM ghcr.io/ba-pherd/platform/task-base:main

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.in,target=/app/requirements.in \
    pip-compile requirements.in && pip-sync

COPY . /app

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
RUN pip install torch_geometric

CMD ["python", "entry_point.py"]