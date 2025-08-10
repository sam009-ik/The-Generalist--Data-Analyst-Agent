FROM mcr.microsoft.com/playwright/python:v1.54.0-jammy

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# install your python libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install browsers
RUN python -m playwright install chromium

# copy app code
COPY . .

# where your temp/session DBs live
ENV SESSION_ROOT=/app/_session_sql
RUN mkdir -p /app/_session_sql

EXPOSE 8000
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
