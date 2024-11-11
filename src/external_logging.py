import subprocess
from loguru import logger
import os

if not os.path.exists('logs'):
    os.makedirs('logs')

logger.add('logs/external_training_log.log', rotation='1 MB', level='INFO', compression='zip')  # Log rotation and compression

process = subprocess.Popen(['python', 'src/train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for line in process.stdout:
    decoded_line = line.decode('utf-8').strip()
    logger.info(decoded_line) 

for line in process.stderr:
    decoded_line = line.decode('utf-8').strip()
    logger.error(decoded_line)  

process.wait()
