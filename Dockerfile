FROM python:3.12.7-slim

# Get build-essentials for annoy lib & curl for healthcheck
RUN apt-get update && apt-get -y install build-essential curl

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]