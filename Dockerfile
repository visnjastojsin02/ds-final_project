FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/train/train.py src/train/train.py

# Default command (arguments passed at runtime)
ENTRYPOINT ["python", "src/train/train.py"]