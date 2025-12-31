# Use Debian Bookworm directly to avoid conflicts with /usr/local python
FROM debian:bookworm-slim

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies including python, gdal, and saga
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-rasterio \
    python3-gdal \
    python3-pytest \
    saga \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any remaining python dependencies
# We use --break-system-packages because we are in a container and want to layer on top of system packages
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Define the entrypoint
ENTRYPOINT ["python3", "dsm2dtm.py"]
