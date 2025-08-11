# Dockerfile
FROM mcr.microsoft.com/playwright/python:v1.54.0-noble

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# browsers already in the image, but this is fine
RUN python -m playwright install chromium

COPY . .

ENV SESSION_ROOT=/app/_session_sql
RUN mkdir -p /app/_session_sql

EXPOSE 8000
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
