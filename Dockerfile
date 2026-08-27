FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY knowledge ./knowledge
COPY fixtures ./fixtures
COPY eval ./eval
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "incident_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

