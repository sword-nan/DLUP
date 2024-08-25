from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Sequence

from utils.transform import cal_time
from .utils import TaskModeSelection, ExcutorModelSelection, process_futures, one_task, multi_task

class TaskPoolExecutor:
    def __init__(self, excutor_mode, task_mode: TaskModeSelection, max_process: int, descrip_state: str, excep_func: Callable, tasks: Callable, args_seq: Sequence[Sequence]) -> None:
        self.tasks = tasks
        self.args_seq = args_seq
        self.excep_func = excep_func
        self.task_mode = task_mode
        self.excutor_mode = excutor_mode
        self.max_process = max_process
        self.descrip_state = descrip_state
        self.select_task_mode()
        self.select_excutor_mode()

    def select_task_mode(self):
        if self.task_mode == TaskModeSelection.ONE_TASK:
            self.get_task_func_args = one_task
        elif self.task_mode == TaskModeSelection.MULTI_TASK:
            self.get_task_func_args = multi_task

    def select_excutor_mode(self):
        if self.excutor_mode == ExcutorModelSelection.THREAD:
            self.excutor = ThreadPoolExecutor
        elif self.excutor_mode == ExcutorModelSelection.PROCESS:
            self.excutor = ProcessPoolExecutor

    def start(self):
        future2args = {}
        @cal_time(self.descrip_state)
        def execute():
            with self.excutor(max_workers=self.max_process) as executor:
                futures = []
                for task, args in self.get_task_func_args(self.tasks, self.args_seq):
                    future = executor.submit(
                        task,
                        *args
                    )
                    futures.append(future)
                    future2args[future] = args
                process_futures(futures, self.excep_func, future2args)
        execute()

    def reset(self):
        pass    