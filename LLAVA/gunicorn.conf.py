import os
NUM_DEVICES = 1
USED_DEVICES = set()

# os.environ["LLAVA_PARAMS_PATH"] = "liuhaotian/llava-v1.6-vicuna-7b" #llava-hf/llava-1.5-7b-hf"  #liuhaotian/llava-v1.6-mistral-7b"  #../llava-weights"

def pre_fork(server, worker):
    # runs on server
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)

def post_fork(server, worker):
    # runs on worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)

def child_exit(server, worker):
    # runs on server
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)

# Gunicorn Configuration
bind = "0.0.0.0:8000"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120
