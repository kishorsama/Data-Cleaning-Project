#python image
FROM python:3.11-slim

#working directory inside container
WORKDIR /app

#copying requirements.txt from local machine to container
COPY requirements.txt .

#installing libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#copying rest files
COPY . .

#running it
CMD [ "python", "demo.py" ]
