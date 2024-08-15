# Import necessary modules
import logging
import os
from datetime import datetime

# Generate a unique log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the path for the logs directory and log file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# Ensure the 'logs' directory exists; create it if it does not
os.makedirs(logs_path, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the file where logs will be written
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Format for log messages
    level=logging.INFO,  # Set the logging level to INFO
)
