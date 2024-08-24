from typing import Dict, List, Tuple, Any

from numba import jit
import numpy as np
import numpy.typing as npt

def unique(x: npt.NDArray):
    """
        返回无重复元素的序列
    """
    unique_arr, counts = np.unique(x, return_counts=True)
    return unique_arr[counts == 1]

@jit(nopython=True)
def tranverse(peptideSpectrumMzs: npt.NDArray, tol: int):
    """
        寻找特异峰

        主要是将所有肽段的 mz 进行 tol 长度的偏移, 偏移的过程中将其记录其是否合并的标志位

        最后筛选出没有进行合并的 mz 就是特异峰离子的 mz
    """
    delta = tol * 1e-6
    featured_ions_flags = np.array(
        [True for _ in range(peptideSpectrumMzs.shape[0])])

    while True:
        # print(len(featured_ions_flags))
        mergeSpectrumMzsInsertIndex = np.searchsorted(
            peptideSpectrumMzs + delta * peptideSpectrumMzs,
            peptideSpectrumMzs,
            side='left')
        a = np.unique(mergeSpectrumMzsInsertIndex)
        if len(a) == len(peptideSpectrumMzs):
            break
        featured_ions_flags = featured_ions_flags[a]
        peptideSpectrumMzsAfterMerge = peptideSpectrumMzs[a]
        for i, seq_index in enumerate(a):
            n = 0
            for index in mergeSpectrumMzsInsertIndex:
                if index == seq_index:
                    n += 1
            if n > 1:
                featured_ions_flags[i] = False
        peptideSpectrumMzs = peptideSpectrumMzsAfterMerge
    return featured_ions_flags, peptideSpectrumMzs

def cal_featured_ions(library: Dict):
    featured_ions_num = 0
    for info in library.values():
        featured_ions_num += len(np.where(info['FeaturedIons'] == 1)[0])
    return len(library), featured_ions_num

def search_featured_ions(library_splitby_window: Dict, tol: int):
    peptideSpectrumMzs = np.array([])
    for _, info in library_splitby_window.items():
        peptideSpectrumMzs = np.append(peptideSpectrumMzs, info['Spectrum'][:, 0])

    peptideSpectrumMzs = unique(peptideSpectrumMzs)
    # print("Merge Before")
    # print(len(peptideSpectrumMzs))
    # start = time.time()
    featured_flags, peptideSpectrumMzsAfterMerge = tranverse(
        peptideSpectrumMzs, tol)
    # print("Merge After")
    # print(len(peptideSpectrumMzsAfterMerge))
    # print(len(featured_flags))
    # featured ions
    peptideSpectrumMzs = peptideSpectrumMzsAfterMerge[featured_flags]
    # print('Featured ions')
    # print(peptideSpectrumMzs)
    for _, info in library_splitby_window.items():
        mz = info['Spectrum'][:, 0]
        featured_ions = np.zeros_like(mz)
        featured_ions_index = np.where(np.in1d(mz, peptideSpectrumMzs))[0]
        featured_ions[featured_ions_index] = 1
        info['FeaturedIons'] = featured_ions
    # print(library_splitby_window)
    # cal_featured_ions(library_splitby_window)
    # end = time.time()
    # print((end - start), 's')
    # print(featured_flags)
    # print(peptideSpectrumMzsAfterMerge)
    # print("After Merge")
    # print(len(peptideSpectrumMzsAfterMerge))

def cal_featured_ions_library(library: Dict):
    modified_peptide_sum, featured_ions_sum = 0, 0
    for _, library_split_by_window in library.items():
        modified_peptide_num, featured_ions_num = cal_featured_ions(
            library_split_by_window)
        modified_peptide_sum += modified_peptide_num
        featured_ions_sum += featured_ions_num
    print('modified_peptide_num: ', modified_peptide_sum)
    print('featured ions num: ', featured_ions_sum)

def divideLibraryByWindows(
    spectraLibraryPath: str,
    windows: List[Tuple[int, ...]],
    tol: int,
    if_featured_ions: bool
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
        将图谱库中的肽段根据窗口 scan 的范围进行划分

        图谱库中包含了肽段的前体离子的 MZ, MZ 在扫描碎裂的窗口的 MZ 范围内说明肽段的前体离子在该窗口内进行碎裂

        因为我们预处理的质谱数据也是根据窗口进行划分的, 这样做方便我们后续进行图谱和肽段的匹配操作

        ### Input Parameters:
        -   spectraLibraryPath: 图谱库的文件路径
        -   windows: 质谱文件划分的窗口

        ### Return:
        -   dividedLibrary: 经质谱数据窗口划分后的图谱库
    """
    library = np.load(spectraLibraryPath, allow_pickle=True).item()
    dividedLibrary = {}
    for window in windows:
        dividedLibrary[window] = {
            key: value
            for key, value in library.items()
            if window[0] <= value["PrecursorMz"] < window[1]
        }
    if if_featured_ions:
        for window in windows:
            search_featured_ions(dividedLibrary[window], tol)
        cal_featured_ions_library(dividedLibrary)
    return dividedLibrary