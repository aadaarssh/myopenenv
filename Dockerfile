FROM python:3.10-slim

WORKDIR /app

# Ensure curl is available for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set necessary environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# The standard command to run an openenv server on HF spaces
CMD ["python", "-m", "openenv.cli", "serve", "--host", "0.0.0.0", "--port", "7860"]
