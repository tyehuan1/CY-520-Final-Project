FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and artifacts
COPY config.py /app/config.py
COPY src/ /app/src/
COPY models/ /app/models/
COPY cache/v2/ /app/cache/v2/
COPY cache/vocabulary.json /app/cache/vocabulary.json

EXPOSE 7860

CMD ["python", "-m", "src.deployment.app"]
