FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY neogen/ neogen/
COPY models/ models/
COPY data/ data/
COPY visualization/ visualization/

# Set environment variables
ENV PORT=8000
ENV API_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "neogen.api.predict:app", "--host", "0.0.0.0", "--port", "8000"]
