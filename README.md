# Simple Classification Service

This is a FastAPI application for an inference service that predicts classes from an input image. It also includes health-related endpoints to check the service status.

## Installation

1. Clone the repository:

```bash
git clone ssh://git@gitlab.deepschool.ru:30022/cvr-aug23/a.gordeev/hw-02-service.git
```

2. Navigate to the cloned directory:

```bash
cd hw-02-service
```

3. Install the required packages:

```bash
pip install -r requirements.txt
pip install .
```

## Usage

To up the service, use the following command:

```bash
uvicorn --host 0.0.0.0 --port $PORT src.app:app
```

## Docker
To use the Docker container for this project, follow these instructions:

1. Build the Docker image:

```bash
docker build -t hw02service:latest .
```

2. Run a Docker container:

```bash
docker run -itd hw02service:latest
```

## API Structure

### /classifier
This prefix groups the endpoint related to classification tasks.

#### POST /classifier/predict
This endpoint allows you to make a prediction based on the given image.

**Input:**
image (bytes): The image file in bytes to make predictions on.

**Output:**
A JSON object containing the key 'classes', which is a list of predicted classes.

### /health
This prefix groups the endpoints related to health checks.

#### GET /health/ping
This endpoint returns a pong message.

**Output:**
A string 🏓 pong!.
Example Request:

#### GET /health/health_checker
This endpoint checks if the service is responding.

**Output:**
An HTTP response with a 200 status code.

## TESTS
Tests can be run locally only
```bash
pytest
```
