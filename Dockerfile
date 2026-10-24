FROM python:3.10-slim

WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Run Flask using Gunicorn bound to port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]