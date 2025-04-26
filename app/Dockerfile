FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_OFFLINE=1 
# ✅ Forces Hugging Face to use local files only

# Install Python and basic dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    apt-get clean

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
