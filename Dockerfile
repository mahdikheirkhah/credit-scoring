# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=2.1.1

# Set the working directory
WORKDIR /app

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y gcc curl libgomp1 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy only the dependency files first to cache them
COPY pyproject.toml poetry.lock ./

# Install project dependencies globally inside the container (no virtualenv needed in Docker)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Copy the rest of your project code into the container
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the Gunicorn server, pointing to the 'server' variable in dashboard.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "results.dashboard.dashboard:server"]