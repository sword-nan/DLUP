import os
from typing import Callable

from multi_task.multi_process import TaskPoolExecutor
from multi_task.utils import TaskModeSelection, ExcutorModelSelection

def multi_process_mzml_tims(num_processes: int, root_path: str, extension_class: str, tol: int, task_func: Callable, excep_func: Callable, *args):
    task = task_func
    args_seq = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(extension_class)]
    args_seq = [(arg, root_path, tol, *args) for arg in args_seq]
    TaskPoolExecutor(
        excutor_mode=ExcutorModelSelection.PROCESS,
        task_mode=TaskModeSelection.ONE_TASK,
        max_process=num_processes,
        descrip_state="所有原始数据文件读取处理完毕",
        excep_func=excep_func,
        tasks=task,
        args_seq=args_seq
    ).start()

    # for path in [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(extension_class)]:
    #     func(path, save_path, tol, *args)
