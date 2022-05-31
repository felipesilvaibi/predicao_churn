
# Use the Python3.7.2 container image
FROM python:3.8.10 

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# run the command to star uWSGI
CMD ["wuwsgi", "app.ini"]