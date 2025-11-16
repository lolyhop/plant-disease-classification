FROM python:3.10-slim

ENV PYTHONPATH=/app
WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY deploy/app/ .

EXPOSE 8000

CMD ["python", "main.py"]
