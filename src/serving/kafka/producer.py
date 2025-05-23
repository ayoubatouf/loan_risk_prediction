import logging
import time
import signal
import sys
from kafka import KafkaProducer
import json
from utils.fake_data import generate_fake_data
from config import KAFKA_BROKER, TOPIC_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    acks="all",
)


def shutdown(signum, frame):
    logger.info("Shutdown signal received. Flushing and closing producer...")
    producer.flush()
    producer.close()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def produce_data():
    logger.info(f"Starting Kafka producer to topic '{TOPIC_NAME}'")
    try:
        while True:
            data = generate_fake_data()
            producer.send(TOPIC_NAME, value=data)
            logger.info(f"Produced message: {data}")
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error in Kafka producer: {e}")
    finally:
        producer.flush()
        producer.close()
        logger.info("Kafka producer closed.")


if __name__ == "__main__":
    produce_data()
