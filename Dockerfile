FROM python:3.7-slim

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Copy application code
COPY . /PhenoBert

# Run PhenoBERT setup
RUN python /PhenoBert/setup.py

WORKDIR /PhenoBert/phenobert/utils/

EXPOSE 8000

CMD ["python", "phenobert_app.py"]