FROM python:3.10-slim

# Setting working directory inside container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/inference/run_inference.py src/inference/run_inference.py


ENTRYPOINT ["python", "src/inference/run_inference.py"]
