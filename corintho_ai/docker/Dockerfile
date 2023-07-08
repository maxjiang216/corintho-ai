# Use the official Python image as the base image
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential python3-dev libblas-dev liblapack-dev gfortran libgsl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt -v

# Copy the remaining application files into the container
COPY . .

# Build the Cython module
RUN python setup.py build_ext --inplace

# Expose the Flask application port
EXPOSE 8080

# Start the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

