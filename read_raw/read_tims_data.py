import os
from typing import Tuple, Dict
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from timspy.dia import TimsPyDIA

from .peak_process import mergePeaks, divideMS2ByWindows
from .concurrency import multi_process_mzml_tims
from utils.transform import cal_time
from utils.io import create_dir

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

def excep_func(e, *args):
    file_path = args[0]
    root_path = args[1]
    error_dir = os.path.join(root_path, "error_data_seq")
    data_name = str(os.path.split(file_path)[-1]).split('.d')[0]
    print(f"{data_name} 文件处理时出现错误: {e}")
    create_dir(error_dir)
    create_dir(os.path.join(error_dir, data_name))

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
    _, file_name = os.path.split(filePath)

    @cal_time(f"{file_name} 文件处理完毕")
    def excute():
        print("start extractint MassSpectrums!")
        merge_path = os.path.join(rootPath, "merge")
        extract_path = os.path.join(rootPath, "extractMs2")
        create_dir(merge_path)
        create_dir(extract_path)
        
        if file_name.replace('.d', '.npy') in os.listdir(merge_path):
            print(f'the file {file_name} has been processed!')
            return
        massSpectrums = extractMs2(filePath)
        save_name = file_name.replace('.d', '.npy')
        savePath = os.path.join(extract_path, save_name)
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
            peaksNumSatifyMassSpectrums, tol)
        savePath = os.path.join(merge_path, save_name)
        np.save(savePath, afterMergedMs2)  # type: ignore
        print("end!")
        del massSpectrums
        del peaksNumSatifyMassSpectrums
        del afterMergedMs2
    excute()

def main(root_path: str, num_processes: int = 20, tol: int = 15):
    multi_process_mzml_tims(
        num_processes=num_processes,
        root_path=root_path,
        extension_class='.d',
        tol=tol,
        task_func=read_tims_data,
        excep_func=excep_func
    )