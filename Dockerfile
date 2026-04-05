# Use a slim Python image to keep the build under 8GB and fast
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if needed (e.g., curl for the validator)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create a non-root user for Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run the FastAPI server
# Important: Bind to 0.0.0.0 so it's reachable externally
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]