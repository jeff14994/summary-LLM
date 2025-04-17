# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install curl and dependencies
RUN apt-get update && apt-get install -y curl

# Set environment variables for Ollama runtime and LLM model
ENV OLLAMA_MODEL=jcai/breeze-7b-32k-instruct-v1_0:q4_0
ENV OUTPUT_DIR=./output
ENV OLLAMA_TIMEOUT=30
ENV OLLAMA_NUM_CTX=1024
ENV OLLAMA_NUM_THREAD=8
ENV MAX_CHUNK_SIZE=500
ENV CHUNK_OVERLAP=100
ENV MAX_WORKERS=4

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama runtime
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose port for the API server
EXPOSE 8000

# Entry point script to start Ollama and pull the model when the container starts
CMD ["bash", "-c", "ollama & sleep 10 && ollama pull jcai/breeze-7b-32k-instruct-v1_0:q4_0 && python3 api_server.py"]

