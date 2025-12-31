# Use an official Python 3.12 runtime as a parent image
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv, our package installer
RUN pip install uv

RUN apt-get update && apt-get install -y libexpat1 libgdal-dev build-essential && rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src/app

# Copy the dependency configuration files
COPY pyproject.toml ./

# Install dependencies using uv
# We install 'test' dependencies as well for running tests in the container
RUN uv pip install --system -e '.[test]'

# Copy the rest of the application source code
COPY . .
