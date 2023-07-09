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
COPY corintho_ai/docker/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt -v

# Copy the remaining application files into the container
COPY corintho_ai/docker ./corintho_ai/docker
COPY corintho_ai/cpp/src/trainmc.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/node.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/game.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/move.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/util.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/include/trainmc.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/node.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/game.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/move.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/util.h ./corintho_ai/cpp/include/
COPY gsl/include/gsl ./gsl/include/gsl

# Build the Cython module
RUN python3 ./corintho_ai/docker/setup.py build_ext --inplace

# Expose the Flask application port
EXPOSE 8080

# Start the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

