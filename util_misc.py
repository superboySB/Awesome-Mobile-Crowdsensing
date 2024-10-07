import subprocess
import os
import json
import logging

# create console handler and set level to debug
logHandler = logging.StreamHandler()
logHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(logHandler)


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)


def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    # logger.info(f'Freest GPU: {freest_gpu["index"]}')
    return freest_gpu['index']
