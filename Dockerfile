# ── Build Stage ───────────────────────────────────────────────────────────────
# Use official slim Python 3.12 image to keep the image lightweight
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout/stderr is unbuffered (important for Docker logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set the working directory inside the container
WORKDIR /app

# ── Install Dependencies ───────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer caching.
# If requirements.txt hasn't changed, this layer won't be rebuilt.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ──────────────────────────────────────────────────────
COPY . .

# Change ownership of all files to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# ── Runtime ────────────────────────────────────────────────────────────────────
# Expose the port that uvicorn will listen on
EXPOSE 8000

# Start FastAPI with uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --workers 1 is fine for MVP; scale this with gunicorn in production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
