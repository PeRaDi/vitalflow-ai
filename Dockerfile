# Use official Python base image with CUDA if you need GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3.10 python3.10-distutils git curl && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install -r ./requirements.txt

# Copy application files
COPY . .

# Set default command
ENTRYPOINT ["python3", "main.py"]

# Allow overriding node_type with ENV at runtime
CMD ["${NODE_TYPE}"]