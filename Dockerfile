FROM python:3.10-slim

# Ensure curl is available for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Set necessary environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# The command to run the FastAPI server on HF spaces
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
