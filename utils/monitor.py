import threading
import subprocess
import psutil
import time
import re


class CPU(threading.Thread):
    def __init__(self, pid, task_completed):
        super().__init__()
        self.pid = pid
        self.task_completed = task_completed
        self.stop_event = threading.Event()
        self.cpu_usage_data = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get the process object using the PID
                process = psutil.Process(self.pid)

                # Check if the process is running
                if not process.is_running():
                    break

                output = subprocess.check_output([
                    'pidstat', '-p', str(self.pid), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                self.cpu_usage_data.append(cpu_)

            except psutil.NoSuchProcess:
                # Process no longer exists, stop monitoring
                break
            except Exception as e:
                print("An error occurred:", e)

            # Check if the task has completed
            if self.task_completed.is_set():
                break
        
        self.stop_monitoring()

    def stop_monitoring(self):
        # Set the stop event to stop hardware monitoring
        self.stop_event.set()


class Memory(threading.Thread):
    def __init__(self, pid, task_completed):
        super().__init__()
        self.pid = pid
        self.task_completed = task_completed
        self.stop_event = threading.Event()
        self.mem_usage_data = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get the process object using the PID
                process = psutil.Process(self.pid)

                # Check if the process is running
                if not process.is_running():
                    break

                output = subprocess.check_output([
                    'pidstat', '-p', str(self.pid), '1', '1', '-r'])
                mem = float(output.splitlines()[-2].split()[-3])
                self.mem_usage_data.append(mem)

            except psutil.NoSuchProcess:
                # Process no longer exists, stop monitoring
                break
            except Exception as e:
                print("An error occurred:", e)

            # Check if the task has completed
            if self.task_completed.is_set():
                break
        
        self.stop_monitoring()

    def stop_monitoring(self):
        # Set the stop event to stop hardware monitoring
        self.stop_event.set()


def _jstat_start():
    subprocess.check_output(
        f'tegrastats --interval 1000 --start --logfile test.txt',
        shell=True)
    time.sleep(2)


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