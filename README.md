# Loan risk prediction

<p align="center">
  <img src="credit.png"/>
</p>

## Overview

This repository hosts a machine learning model that predicts loan risk for banks using LightGBM. Accurately assessing loan risk is crucial for financial institutions to minimize defaults, optimize lending strategies, and ensure regulatory compliance. The model leverages a comprehensive dataset capturing borrower and loan characteristics to deliver reliable credit risk predictions. It can also integrates with Apache Kafka to handle streaming data, enabling real-time risk assessment and continuous scoring for dynamic loan applications. This combination of powerful gradient boosting and scalable streaming makes the system well-suited for modern banks focused on improving risk management and operational efficiency.

## Usage

### Running the ML Pipeline Scripts

Navigate to the `/scripts` directory to execute different parts of the ML pipeline. You can run any of the following scripts depending on your needs:

- `all_scripts.py` — Run the entire pipeline end-to-end

- `data_script.py` — Handle data preprocessing and feature engineering

- `training_script.py` — Train the LightGBM model

- `inference_script.py` — Perform batch inference on new data

- `ddrift_script.py` — Monitor data drift

Example : 
```
cd scripts
python3 training_script.py
```

### Serving the Model via API

Start the FastAPI server using uvicorn:

```
uvicorn src.serving.api.app:app --host 0.0.0.0 --port 8000
```

Test the API with your own data using the provided shell script:

```
./test_api.sh
```

### Running Kafka for Streaming

Start your Kafka broker (make sure Kafka and Zookeeper are installed and running):

```
# Start Zookeeper
zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka broker
kafka-server-start.sh config/server.properties
```
Run the Kafka producer and consumer scripts located in `src/serving/kafka`:

```
python3 src/serving/kafka/producer.py
python3 src/serving/kafka/consumer.py
```
These scripts handle streaming loan data to and from Kafka topics, enabling real-time loan risk prediction.

### Using Docker

Build the Docker image:

```
docker build -t loan-risk-prediction .
```

Run the container exposing port 8000:
```
docker run -p 8000:8000 loan-risk-prediction
```

This will launch the FastAPI server inside the container, making the loan risk prediction API accessible.
