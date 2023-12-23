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

### /detector
This prefix groups the endpoint related to detector tasks.

#### POST /detector/predict
This endpoint allows you to make a prediction based on the given image.

**Input:**
image (bytes): The image file in bytes to make predictions on.

**Output:**
 {"bboxes": [{coords of bbox} for bbox in bboxes]}

#### POST /detector/predict_mask
This endpoint allows you to make a prediction based on the given image.

**Input:**
image (bytes): The image file in bytes to make predictions on.

**Output:**
{"base64_encoded_mask": base64_encoded_mask}

### /recognizer
This prefix groups the endpoint related to recognizer tasks.

#### POST /recognizer/predict_barcode
This endpoint allows you to make a prediction based on the given image of the barcode.

**Input:**
image (bytes): The image of the barcode file in bytes to make predictions on.

**Output:**
"predicted barcode info."


#### POST /recognizer/predict_image
This endpoint allows you to make a prediction based on the given image.
It will run the detector model and then the recognizer model will be applied to each barcode detected.

**Input:**
image (bytes): The image file in bytes to make predictions on.

**Output:**
dict of barcode's coords and its info.

### /health
This prefix groups the endpoints related to health checks.

#### GET /health/health_checker
This endpoint checks if the service is responding.

**Output:**
An HTTP response with a 200 status code.

## TESTS
Tests can be run locally only
```bash
pytest
```
