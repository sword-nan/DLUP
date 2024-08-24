from typing import List, Tuple, Dict

import numpy as np
import numpy.typing as npt
from numba import jit

@jit(nopython=True)
def tranversePeaks(peaks: npt.NDArray, tol: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    if the |mz1 - mz2| < tol * mz1, we merge the peaks.

    the new peak will be as belows:
    -   mz = mz1
    -   intensity = (intensity1 + intensity2) / 2

    if many peaks satisfy, we merge all peaks, and the result will be
    -   mz = mz1
    -   intensity = sum_{i = 1}^{n} intensity_i / n

    #### Input Parameters:
    -   `massSpectrums`: the mass sptctrums we read from mzml file
    -   `tol`: If the distance between (m/z) of peaks is within a tolerance of * e-6, we will merge the peaks.
    the mz of the new peak will be left mz, and the intensity will be average intensities of the peaks 

    #### Return:
    -   `mergedSpectrumMzs`: the mz of peaks after merge
    -   `mergedSpectrumIntensities`: the intensities of peaks after merge
    """
    mergeSpectrumMzsInsertIndex = np.searchsorted(
        peaks[:, 0] + tol * peaks[:, 0], peaks[:, 0])
    a = np.unique(mergeSpectrumMzsInsertIndex)
    mergedSpectrumMzs = peaks[a, 0]
    mergedSpectrumIntensities = []
    for i in a:
        m = 0.0
        n = 0
        for kk in range(len(mergeSpectrumMzsInsertIndex)):
            if mergeSpectrumMzsInsertIndex[kk] == i:
                m += peaks[kk, 1]
                n += 1
        m = m / n
        mergedSpectrumIntensities.append(m)
    return mergedSpectrumMzs, np.array(mergedSpectrumIntensities)

def mergePeaks(peaks: npt.NDArray, tol: int):
    """
    #### Input Parameters:
    -   `massSpectrums`: the mass sptctrums we read from mzml file
    -   `tol`: If the distance between (m/z) of peaks is within a tolerance of * e-6, we will merge the peaks.
    the mz of the new peak will be left mz, and the intensity will be average intensities of the peaks 

    #### Return:
    -   `peaks`: new peaks after merge
    """
    tol = tol * 1e-6  # type: ignore
    peakNums = len(peaks)  # type: ignore
    while True:
        mergedSpectrumMzs, mergedSpectrumIntensities = tranversePeaks(
            peaks, tol)
        peaks = np.transpose(
            np.array((mergedSpectrumMzs, mergedSpectrumIntensities)))  # type: ignore
        if peakNums == len(peaks):  # type: ignore
            break
        peakNums = len(peaks)  # type: ignore
    return peaks

def divideMS2ByWindows(massSpectrums: npt.NDArray, windows: List[Tuple[int, int]]) -> Dict[Tuple[int, int], npt.NDArray]:
    """
    #### Input Parameters: 
    -    `massSpectrums`: the mass sptctrums we read from mzml file
    -    `windows`: the scan window we have deduplicated

    #### Return:
    -   DividedMS2: the mass spectrums divided by windows
    """
    DividedMS2: Dict[Tuple[int, int], npt.NDArray] = {}
    for window in windows:
        DividedMS2[window] = [
            ms2 for ms2 in massSpectrums if ms2[1] == window]  # type: ignore
        DividedMS2[window] = np.array(DividedMS2[window], dtype=object)
        # Sort by retention time
        DividedMS2[window] = DividedMS2[window][DividedMS2[window]
                                                [:, 2].argsort()]
    return DividedMS2
