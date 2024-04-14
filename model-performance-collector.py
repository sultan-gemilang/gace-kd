import pathlib
import importlib
import argparse
import time
import threading
import subprocess
import os
import re
from datetime import datetime
from timeit import default_timer as timer

import silence_tensorflow.auto

import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

import tensorflow as tf
import tensorflow_datasets as tfds

# Global config
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)


# Threads
class CPU(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(cpu_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


class Memory(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output([
                    'pidstat', '-p', str(os.getpid()), '1', '1', '-r'])
                mem_ = float(output.splitlines()[-2].split()[-3])
                self._list.append(mem_)

            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list, output
        except:
            self.result = 0, self._list, output

    def stop(self):
        self.event.set()


def _jstat_start():
    subprocess.check_output(
        f'tegrastats --interval 1000 --start --logfile test.txt',
        shell=True)


def _jstat_stop():
    subprocess.check_output(f'tegrastats --stop', shell=True)
    out = open("test.txt", 'r')
    lines = out.read().split('\n')
    entire = []
    try:
        for line in lines:
            pattern = r"GR3D_FREQ (\d+)%"
            match = re.search(pattern, line)
            if match:
                gpu_ = match.group(1)
                entire.append(float(gpu_))
        # entire = [num for num in entire if num > 10.0]
        result = sum(entire) / len(entire)
    except:
        result = 0
        entire = entire
        pass

    subprocess.check_output("rm test.txt", shell=True)
    return result, entire


def prepare_dataset(batch):
    IMG_SIZE = (224, 224)

    image_dir = f"{pathlib.Path.cwd()}/assets/"
    image_dataset = tf.keras.utils.image_dataset_from_directory(image_dir,
                                                                image_size=IMG_SIZE,
                                                                batch_size=batch)

    return image_dataset


def calc_latency_per_batch(inference_time, batch):
    latency_s = (inference_time * batch) / 2500  # Latency in seconds!
    latency_ms = round(latency_s * 1000, 5)  # In miliseconds!

    return latency_s, latency_ms


def do_inference(path: str, test_set, batch):
    load_engine = EngineFromBytes(BytesFromPath(path))

    with TrtRunner(load_engine) as runner:
        # Threading
        cpu_thread = CPU()
        mem_thread = Memory()
        cpu_thread.start()
        mem_thread.start()
        _jstat_start()

        begin = datetime.now()
        for image, label in test_set:
            outputs = runner.infer(feed_dict={'input_1': image.numpy()})
            inferred = np.argmax(outputs['dense'])

        gpu = float(_jstat_stop()[0])
        end = datetime.now()
        delta = end - begin

        cpu_thread.stop()
        mem_thread.stop()
        cpu_thread.join()
        mem_thread.join()

        cpu_use = round(cpu_thread.result[0], 2)
        mem_use = round(mem_thread.result[0] / 1024, 2)
        gpu = round(gpu, 2)

        latency_s, latency_ms = calc_latency_per_batch(delta.total_seconds(), batch) 
        print(f"{delta.total_seconds()},{latency_ms},{cpu_use},{gpu},{mem_use}")


def main(args):
    # clear cache
    os.system("echo 'CloudLab12#$%' | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")

    test_set = prepare_dataset(batch=1)
    
    for _ in range(35):
        do_inference(args.engine, test_set, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--engine')

    args = parser.parse_args()

    main(args)
