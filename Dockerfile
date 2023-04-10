FROM python:3.10.10

WORKDIR /dsm2dtm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    saga \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY requirements_dev.txt .
RUN pip install --no-cache-dir --upgrade -r requirements_dev.txt

COPY ./src ./src
COPY ./tests ./tests
COPY ./data ./data

CMD [ "pytest"]
