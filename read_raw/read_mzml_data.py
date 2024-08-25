import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from pymzml.run import Reader

from .concurrency import multi_process_mzml_tims
from .peak_process import mergePeaks, divideMS2ByWindows
from utils.io import create_dir
from utils.transform import cal_time

def loadMS2(path: str) -> npt.NDArray:
    """
        use the pymzml package to read the MS2 data

        the file that must be what end with `.mzML`

    #### Input Paratmeters:
    -   `path`: the mzML file path

    #### Return:
    -    `MS2`: the mass spectrum with [array([mz, intensity]), scan window, RT, index]
    """
    diaMassSpectrum = Reader(path)
    MS2 = [
        [
            spectrum.peaks('raw'),  # 原始二级质谱数据
            (  # Tuple(MZ1, MZ2)
                spectrum['MS:1000827'] - \
                spectrum['MS:1000828'],  # 窗口左端 MZ 值
                spectrum['MS:1000827'] + \
                spectrum['MS:1000829']  # 窗口右端 MZ 值
            ),
            spectrum['MS:1000016'],  # 质谱图的保留时间
            i
        ]
        for i, spectrum in tqdm(enumerate(diaMassSpectrum), path)
        if spectrum.ms_level == 2.0
    ]
    return np.array(MS2, dtype=object)

def excep_func(e, *args):
    file_path = args[0]
    root_path = args[1]
    error_dir = os.path.join(root_path, "error_data_seq")
    data_name = str(os.path.split(file_path)[-1]).split('.mzML')[0]
    print(f"{data_name} 文件处理时出现错误: {e}")
    create_dir(error_dir)
    create_dir(os.path.join(error_dir, data_name))

def readMzmlData(path: str, root_path: str, tol: int):
    _, file_name = os.path.split(path)

    @cal_time(f"{file_name} 文件处理完毕")
    def excute():
        save_path = os.path.join(root_path, 'merge')
        MS2 = loadMS2(path)
        for i in range(len(MS2)):
            MS2[i][0] = mergePeaks(MS2[i][0], tol)
        windows = np.array(list(set(MS2[:, 1])))
        windows = windows[np.argsort(windows[:, 0])]
        windows = [tuple(window) for window in windows]
        dividedMS2 = divideMS2ByWindows(MS2, windows)  # type: ignore
        save_name = file_name.replace('.mzML', '.npy')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, save_name), dividedMS2)  # type: ignore
    excute()

def main(root_path: str, num_processes: int = 20, tol: int = 15):
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.mzML',
        tol=tol,
        task_func=readMzmlData,
        excep_func=excep_func
    )