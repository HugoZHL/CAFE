import concurrent.futures
import subprocess
import json
import os
import os.path as osp
import sys
import torch
from time import sleep

cur_dir = osp.split(osp.abspath(__file__))[0]

# Simulate a task that uses a GPU
def run_task(task, gpu_id):
    need_wrap_path = ['data_path', 'tensor_board_filename']
    for path in need_wrap_path:
        task[path] = osp.join(cur_dir, task[path])
    log_dir = task['tensor_board_filename']
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, 'stdouterr.log')
    task_name = osp.split(log_dir)[1]
    print(f"Running task {task_name} on GPU {gpu_id}...")

    with open(osp.join(log_dir, 'config.json'), 'w') as fw:
        json.dump(task, fw, indent=4)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Open the log file
    with open(log_file, 'w') as log:
        command = ["python", osp.join(cur_dir, 'main.py')]
        for k, v in task.items():
            command += [f'--{k}', str(v)]
        result = subprocess.run(command, stdout=log, stderr=log, text=True, env=env)

    print(f"Task {task_name} on GPU {gpu_id} finished with return code {result.returncode}")

def load_tasks(config_file, flatten=['compress_rate', 'cafe_sketch_threshold', 'cafe_hash_rate', 'cafe_decay']):
    with open(config_file, 'r') as file:
        config = json.load(file)
    base_args = config['base']
    tasks = []
    methods = ['full', 'hash', 'qr', 'ada', 'mde', 'cafe', 'off']
    for met in methods:
        if met in config:
            new_task = base_args.copy()
            flags = {}
            for k, v in config[met].items():
                if k not in flatten or not isinstance(v, list):
                    new_task[k] = v
                else:
                    flags[k] = v
            if flags == {}:
                tasks.append(new_task)
                continue
            fls = list(flags.keys())
            if 'compress_rate' in flags:
                diff = 'compress_rate'
            else:
                if len(flags) > 1:
                    raise AssertionError("Not support more than 1 difference besides compress_rate!")
                diff = fls[0]
            for vs in zip(*flags.values()):
                cur_new_task = new_task.copy()
                for fl, v in zip(fls, vs):
                    cur_new_task[fl] = v
                cur_new_task['tensor_board_filename'] += str(cur_new_task[diff])
                tasks.append(cur_new_task)
    return tasks

def schedule(config_files):
    num_gpus = torch.cuda.device_count()  # Number of GPUs available
    tasks = []
    for config_file in config_files:
        tasks.extend(load_tasks(config_file))
    print('Number of tasks:', len(tasks))

    # Use ThreadPoolExecutor to manage GPU tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        # Dictionary to keep track of which GPU is running which task
        gpu_to_task = {}

        # Submit initial tasks to GPUs
        for gpu_id in range(num_gpus):
            if tasks:
                task = tasks.pop(0)
                future = executor.submit(run_task, task, gpu_id)
                gpu_to_task[future] = gpu_id

        # Process completed tasks and submit new ones
        while tasks:
            done, _ = concurrent.futures.wait(gpu_to_task.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                gpu_id = gpu_to_task.pop(future)
                if tasks:
                    task = tasks.pop(0)
                    future = executor.submit(run_task, task, gpu_id)
                    gpu_to_task[future] = gpu_id
            sleep(60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise AssertionError("Usage: python job_scheduler.py <config_file>")
    config_files = sys.argv[1:]
    schedule(config_files)
