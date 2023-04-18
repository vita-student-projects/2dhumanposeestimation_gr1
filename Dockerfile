FROM python:3.9

WORKDIR /usr/src

# Install project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt