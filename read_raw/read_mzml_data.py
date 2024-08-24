import os
import time

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from pymzml.run import Reader

from .multi_process import multi_process_mzml_tims
from .peak_process import mergePeaks, divideMS2ByWindows
from utils.transform import transform_time

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

def readMzmlData(path: str, root_path: str, tol: int):
    start = time.time()
    save_path = os.path.join(root_path, 'merge')
    MS2 = loadMS2(path)
    for i in range(len(MS2)):
        MS2[i][0] = mergePeaks(MS2[i][0], tol)
    windows = np.array(list(set(MS2[:, 1])))
    windows = windows[np.argsort(windows[:, 0])]
    windows = [tuple(window) for window in windows]
    dividedMS2 = divideMS2ByWindows(MS2, windows)  # type: ignore
    _, file_name = os.path.split(path)
    save_name = file_name.replace('.mzML', '.npy')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, save_name), dividedMS2)  # type: ignore
    end = time.time()
    print('-'*120)
    hour, minute, second = transform_time(end - start)
    print('{} 文件处理完毕，耗时 {:.1f}h {:.1f}m {:.1f}s'.format(file_name, hour, minute, second))
    print('-'*120)

def main(root_path: str, num_processes: int = 20, tol: int = 15):
    start = time.time()
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.mzML',
        tol=tol,
        func=readMzmlData
    )
    end = time.time()
    hour, minute, second = transform_time(end - start)
    print('-'*120)
    print('所有文件处理完毕，耗时 {:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))
    print('-'*120)

if __name__ == "__main__":

    start = time.time()
    for root_path in [
        '/data/xp/data/astral_20231016_300ngPlasmaSample',
        '/data/xp/data/lfq32',
        '/data/xp/data/NoRT_LFQ',
        '/data/xp/data/Hela'
    ]:
        main(root_path, 15, 15)
    end = time.time()
    hour, minute, second = transform_time(end - start)
    print('所有数据处理完毕，耗时 {:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))