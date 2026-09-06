import logging
import sys

def setup_logger(name, level=logging.DEBUG):
  logger = logging.getLogger(name)
  logger.setLevel(level)

  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(logging.INFO)

  formatter = logging.Formatter(
        '\n%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
  
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  return logger  