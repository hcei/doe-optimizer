FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

ENTRYPOINT ["python", "doe_optimizer.py"]
CMD ["--budget", "30", "--seed", "42"]
