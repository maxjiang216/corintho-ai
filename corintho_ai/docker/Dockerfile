# Stage 1: Build
FROM python:3.9-slim as builder

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY corintho_ai/docker/requirements.txt .

# Copy the remaining application files into the container
COPY corintho_ai/docker ./corintho_ai/docker
COPY corintho_ai/cpp/src/dockermc.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/trainmc.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/node.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/game.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/move.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/src/util.cpp ./corintho_ai/cpp/src/
COPY corintho_ai/cpp/include/dockermc.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/trainmc.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/node.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/game.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/move.h ./corintho_ai/cpp/include/
COPY corintho_ai/cpp/include/util.h ./corintho_ai/cpp/include/
COPY gsl/include/gsl ./gsl/include/gsl

# Install build dependencies and build Cython module
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential python3-dev libblas-dev liblapack-dev libgsl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt -v && \
    cd ./corintho_ai/docker && python3 setup.py build_ext --inplace && cd ../..

# Stage 2: Run
FROM python:3.9-alpine

COPY --from=builder /app /app

# Set the working directory
WORKDIR /app

# Install Flask in the final stage
RUN pip install Flask Flask-CORS

# Set FLASK_APP environment variable
ENV FLASK_APP=/app/corintho_ai/docker/app.py

# Expose the Flask application port
EXPOSE 8080

# Start the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
