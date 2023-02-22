FROM tiangolo/uvicorn-gunicorn:python3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
