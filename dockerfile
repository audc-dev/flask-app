# Example Dockerfile structure
FROM python:3.12

WORKDIR /flask-app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code
COPY app.py .
COPY templates/ templates/

# Copy the model directory
COPY model/ model/

# Copy the upload directory
COPY uploads/ uploads/

# Copy de YAML file
COPY deployment.yaml .

# Make sure model directory exists
RUN mkdir -p model

EXPOSE 5000

CMD ["python", "/flask-app/app.py"]