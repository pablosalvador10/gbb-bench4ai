# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY src /app/src
COPY my_utils/ml_logging.py /app/my_utils/ml_logging.py 
COPY my_utils/__init__.py /app/my_utils/__init__.py  
COPY my_utils/data/evaluations/dataframe/golden_eval_dataset.csv /app/my_utils/data/evaluations/dataframe/golden_eval_dataset.csv
COPY my_utils/images/ /app/my_utils/images/
COPY requirements.txt /app/requirements.txt

RUN mkdir -p /app/.streamlit
COPY src/app/settings/config.toml /app/.streamlit/config.toml
RUN pip install --no-cache-dir -r requirements.txt
ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml

EXPOSE 8501

# Define the command to run the Streamlit application
ENTRYPOINT ["streamlit", "run"]
CMD ["src/app/Home.py"]