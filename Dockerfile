FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model bundle
COPY src/*.py ./
COPY models ./models
COPY models /models

# Expose the FastAPI port
EXPOSE 8001

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

