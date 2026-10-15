# Build from the repo root:  docker compose up --build   (compose file is in r8_app/)
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 RAG_REPO_PATH=/srv
COPY r8_app/requirements.txt r8_app/requirements-rag.txt ./r8_app/
RUN pip install --no-cache-dir -r r8_app/requirements.txt -r r8_app/requirements-rag.txt
COPY . .
WORKDIR /srv/r8_app
ENV PYTHONPATH=/srv/r8_app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
