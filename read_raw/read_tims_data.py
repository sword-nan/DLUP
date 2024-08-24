import os
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Dict
import time

import numpy as np
import numpy.typing as npt
from timspy.dia import TimsPyDIA

from utils.io import create_dir
from .peak_process import mergePeaks, divideMS2ByWindows
from utils.transform import transform_time
from read_raw.multi_process import multi_process_mzml_tims

def extractMs2(path: str):
    """
        读取 `.d` 文件中的质谱数据, 将其中的二级质谱数据提取出来

        最后得到如下的数据格式
        [
            # 峰    扫描的窗口   保留时间 frame标号 scan标号  淌度信息
            [peaks, scanWindow, frameRT, frameID, scanID, ionMobity],
            ...
        ]

    """
    D = TimsPyDIA(path)
    winGroupScan: Dict[int, Dict[Tuple[int, int], Tuple]] = defaultdict(dict)
    # [WindowGroup, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy, mz_left, mz_right]
    for windows_id, scan_left, scan_right, mz_left, mz_right in D.windows[['WindowGroup', 'ScanNumBegin', 'ScanNumEnd', 'mz_left', 'mz_right']].values:
        winGroupScan[int(windows_id)][(int(scan_left), int(scan_right))] = (
            mz_left, mz_right)
    massSpectrums = []

    for frameID in D.ms2_frames:
    # for frameID in tqdm(D.ms2_frames, total=len(D.ms2_frames)):
        # [frame, scan, tof, intensity, mz, inv_ion_mobility, retention_time]
        ms2FrameData = D.query(frames=frameID)
        frameRT = ms2FrameData['retention_time'][0]
        # key: scanID, value: [peaks, inv_ion_mobility]
        scanDataDict: Dict[int, list] = {}
        for scan, mz, intensity, ion_mobility in ms2FrameData[['scan', 'mz', 'intensity', 'inv_ion_mobility']].values:
            scan_id = int(scan)
            peak = (mz, intensity)
            if scan_id not in scanDataDict.keys():
                scanDataDict[scan_id] = [np.array([peak]), ion_mobility]
            else:  # 将相同 scan 下的峰整合为一组峰
                scanDataDict[scan_id][0] = np.vstack(
                    (scanDataDict[scan_id][0], peak))
        # end for scanData
        # 找到 frame 对应的 window
        # 如果 scanID 在 window 对应的 scan 范围内, 则将其二级质谱数据保留
        frameToWindow = (frameID - 1) % (max(winGroupScan.keys()) + 1)

        for scanID, massSpectrum in scanDataDict.items():
            peaks = massSpectrum[0]
            peaks = peaks[np.argsort(peaks[:, 0])]
            ionMobity = massSpectrum[1]
            for scanRange, scanWindow in winGroupScan[frameToWindow].items():
                if scanRange[0] <= scanID <= scanRange[1]:
                    massSpectrums.append(
                        [peaks, scanWindow, frameRT, frameID, scanID, ionMobity])
                    break

    array_massSpectrums = np.array(massSpectrums, dtype=object)
    del massSpectrums
    del D
    return array_massSpectrums

def processPeaks(
    massSpectrums: npt.NDArray,
    file: str,
    tol: int
    ):
    """
        处理二级质谱的峰

        返回峰合并操作之后的质谱数据
    """
    for i in range(len(massSpectrums)):
        massSpectrums[i][0] = mergePeaks(massSpectrums[i][0], tol)
    windows = np.array(list(set(massSpectrums[:, 1])))
    windows = windows[np.argsort(windows[:, 0])]
    windows = [tuple(window) for window in windows]
    print("divide the Ms2 by scan windows!")
    dividedMs2 = divideMS2ByWindows(
        massSpectrums, windows)  # type: ignore
    return dividedMs2

def read_tims_data(filePath: str, rootPath: str, tol: int):
    """
        读取 tims 数据

        从 .d 文件数据中提取二级质谱数据并进行峰合并操作

        Parameters
        ---
        -   filePath: .d 文件所在文件路径
        -   root_path: 保存提取后的数据根目录
        -   
    """
    print("start extractint MassSpectrums!")
    merge_path = os.path.join(rootPath, "merge")
    extract_path = os.path.join(rootPath, "extractMs2")
    create_dir(merge_path)
    create_dir(extract_path)
    _, file_name = os.path.split(filePath)
    if file_name.replace('.d', '.npy') in os.listdir(merge_path):
        print(f'the file {file_name} has been processed!')
        return
    start = time.time()
    massSpectrums = extractMs2(filePath)
    fileName = file_name.split('.')[0]
    savePath = os.path.join(extract_path, f"{fileName}.npy")
    # np.save(savePath, massSpectrums)
    print("end!")

    peaksNumSatifyMassSpectrums = []
    for massSpectrum in massSpectrums:
        if len(massSpectrum[0]) >= 6:
            peaksNumSatifyMassSpectrums.append(massSpectrum)
    peaksNumSatifyMassSpectrums = np.array(
        peaksNumSatifyMassSpectrums, dtype=object)

    print("start merging Peaks!")
    afterMergedMs2 = processPeaks(
        peaksNumSatifyMassSpectrums, fileName, tol)
    savePath = os.path.join(merge_path, f"{fileName}.npy")
    np.save(savePath, afterMergedMs2)  # type: ignore
    print("end!")
    end = time.time()
    print('-'*120)
    hour, minute, second = transform_time(end - start)
    print('{} 文件处理完毕，耗时 {:.1f} h {:.1f} m {:.1f}'.format(fileName, hour, minute, second))
    print('-'*120)
    del massSpectrums
    del peaksNumSatifyMassSpectrums
    del afterMergedMs2

def main(root_path: str, num_processes: int = 20, tol: int = 15):
    start = time.time()
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.d',
        tol=tol,
        func=read_tims_data
    )
    end = time.time()
    hour, minute, second = transform_time(end - start)
    print('-'*120)
    print('所有文件处理完毕，耗时 {:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))
    print('-'*120)

if __name__ == "__main__":
    start = time.time()
    for root_path in [
        '/data/xp/data/1. T2D-CAD-RJ',
        '/data/xp/data/2. MeS_YN',
        '/data/xp/data/3. brain cohort-JW'
    ]:
        main(root_path, 15, 15)
    end = time.time()
    hour, minute, second = transform_time(end - start)
    print('所有数据处理完毕，耗时 {:.1f}h {:.1f}m {:.1f}s'.format(hour, minute, second))
    # import os
    # tol = 15
    # rootPath = "./"
    # filePaths = [rootPath +
    #              file for file in os.listdir(rootPath) if file.endswith('.d')]
    # for filePath in filePaths:
    #     print("start extractint MassSpectrums!")
    #     massSpectrums = extractMs2(filePath)
    #     fileName = filePath.split('/')[-1].split('.')[0]
    #     savePath = os.path.join(rootPath, "extractMs2", "{fileName}.npy")
    #     np.save(savePath, massSpectrums)
    #     print("end!")

    #     peaksNumSatifyMassSpectrums = []
    #     for massSpectrum in massSpectrums:
    #         if len(massSpectrum[0]) >= 6:
    #             peaksNumSatifyMassSpectrums.append(massSpectrum)
    #     peaksNumSatifyMassSpectrums = np.array(
    #         peaksNumSatifyMassSpectrums, dtype=object)

    #     print("start merging Peaks!")
    #     afterMergedMs2 = processPeaks(
    #         peaksNumSatifyMassSpectrums, fileName, tol)
    #     savePath = os.path.join(rootPath, "merge", "{fileName}.npy")
    #     np.save(savePath, afterMergedMs2)  # type: ignore
    #     print("end!")
