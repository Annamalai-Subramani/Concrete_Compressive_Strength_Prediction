import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


log_path=os.path.join(os.getcwd(),"logs")

os.makedirs(log_path,exist_ok=True)
log_filepath=os.path.join(log_path,LOG_FILE)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == '__main__':
    logging.info("Logging has started")