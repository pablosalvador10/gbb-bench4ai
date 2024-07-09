# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Optionally set the PYTHONPATH if your application requires it
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Add the current directory contents into the container at /app
COPY src /app/src
COPY utils /app/utils
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["src/app/Home.py"]