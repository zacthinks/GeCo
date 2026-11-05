# Cloud Run technical preview for GeCo 0.10.0.  No uploaded research data.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GECO_DEMO_TEMPLATE=/app/demo_data/starter.geco \
    GECO_DEMO_SEMANTIC=0 \
    GECO_DEMO_IDLE_SECONDS=600 \
    PORT=8080

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY demo/ ./demo/

# Install THIS checkout of GeCo, not a stale wheel or removed uv.lock.
# The synthetic project is an explicit infrastructure-only placeholder.
RUN python -m pip install --no-cache-dir . flask gunicorn \
    && python -m demo.build_starter

USER 1000
EXPOSE 8080
CMD ["gunicorn", "--workers", "1", "--threads", "8", \
     "--bind", "0.0.0.0:8080", "--timeout", "240", "demo.cloudrun:app"]
