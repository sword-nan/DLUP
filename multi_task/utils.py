import gc
from enum import Enum
from concurrent.futures import as_completed
from typing import Sequence

class TaskModeSelection(Enum):
    ONE_TASK = 1
    MULTI_TASK = 2

class ExcutorModelSelection(Enum):
    THREAD = 1
    PROCESS = 2

def one_task(task, args_seq: Sequence[Sequence]):
    for args in args_seq:
        yield task, args

def multi_task(task_seq, args_seq: Sequence[Sequence]):
    for task, args in zip(task_seq, args_seq):
        yield task, args

def process_futures(futures, exception_func, future2args):
    for future in as_completed(futures):
        try:
            future.result()
            # 处理结果
        except Exception as e:
            exception_func(e, *future2args[future])
        finally:
            # 显式删除 future 并触发垃圾回收
            futures.remove(future)
            del future
            gc.collect()