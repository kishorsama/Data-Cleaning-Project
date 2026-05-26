#python image
FROM python:3.11-slim

#working directory inside container
WORKDIR /app

#git install
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

#copying requirements.txt from local machine to container
COPY requirements.txt .

#installing libraries from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter

#copying rest files
COPY . .

# Create a build timestamp file inside the container
RUN sh -c 'echo "Container built on: $(date)" > /etc/container_build_version'
#running it
CMD [ "python", "demo.py" ]
