import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

def multi_process_mzml_tims(num_processes: int, root_path: str, extension_class: str, tol: int, func: Callable, *args):

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []

        for path in [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(extension_class)]:
            future = executor.submit(
                func, path, root_path, tol, *args)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
                # 处理结果
            except Exception as e:
                print(f"任务执行时发生错误: {e}")
            finally:
                # 显式删除 future 并触发垃圾回收
                futures.remove(future)
                del future
                gc.collect()

    # for path in [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(extension_class)]:
    #     func(path, save_path, tol, *args)
