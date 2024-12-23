# Use a lightweight base image with Python pre-installed
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PIP_NO_CACHE_DIR=TRUE

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the package and requirements file
COPY ./trainer /app/trainer
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Define the entry point for the container
ENTRYPOINT ["python", "-m", "trainer.task"]