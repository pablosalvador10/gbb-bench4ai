# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY src /app/src
COPY utils /app/utils
COPY requirements.txt /app/requirements.txt

RUN mkdir -p /app/.streamlit
COPY src/app/settings/config.toml /app/.streamlit/config.toml
RUN pip install --no-cache-dir -r requirements.txt
ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml

EXPOSE 8501

# Define the command to run the Streamlit application
ENTRYPOINT ["streamlit", "run"]
CMD ["src/app/Home.py"]