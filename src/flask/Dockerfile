
# Use the Python3.7.2 container image
FROM python:3.8.10 

# Set the working directory to /app
WORKDIR /app

# Install the dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
ADD . /app

# run the command to star uWSGI
CMD ["uwsgi", "--ini", "/app/uwsgi.ini"]