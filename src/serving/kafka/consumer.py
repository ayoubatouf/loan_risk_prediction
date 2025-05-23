import logging
import signal
import sys
import json
import pandas as pd
import joblib
from kafka import KafkaConsumer
from src.pipelines.inference_pipeline import run_inference_pipeline
from config.paths import MODEL_PATH, SCALER_PATH
from config import KAFKA_BROKER, TOPIC_NAME

# cmd kafka broker : /opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    group_id="inference_consumer_group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
)


def shutdown(signum, frame):
    logger.info("Shutdown signal received. Closing Kafka consumer...")
    consumer.close()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def consume_data():
    logger.info(f"Starting Kafka consumer on topic '{TOPIC_NAME}'")
    try:
        for message in consumer:
            input_data = message.value
            logger.info(f"Consumed message: {input_data}")

            input_df = pd.DataFrame([input_data])

            try:
                y_pred, _ = run_inference_pipeline(model, scaler, input_df)
                logger.info(f"Prediction: {y_pred[0]}")
            except Exception as e:
                logger.error(f"Error processing data in inference pipeline: {e}")
    except Exception as e:
        logger.error(f"Error in Kafka consumer: {e}")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed.")


if __name__ == "__main__":
    consume_data()
