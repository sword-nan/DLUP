import os
import time
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Sequence, List, Any, Literal, Callable

import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from numba import njit

from utils.transform import transform_time
from utils.io import create_dir, read_dict_npy
from library.window_partition import divideLibraryByWindows

@njit
def search(spectrum_mz: npt.NDArray, peptide_mzs: npt.NDArray):
    return np.searchsorted(spectrum_mz, peptide_mzs)

class FilterSimilarityFunction:

    @staticmethod
    def cosine(A: npt.NDArray, B: npt.NDArray):
        norm_A = np.linalg.norm(A, axis=1, keepdims=True)
        norm_B = np.linalg.norm(B, axis=1, keepdims=True)
        norlize_A = A / norm_A
        norlize_B = B / norm_B
        return np.dot(norlize_A, norlize_B.T)
    
    @staticmethod
    def MAE(A: npt.NDArray, B: npt.NDArray):
        sum_norm_B = B / np.sum(B, axis=1, keepdims=True)
        scores = np.zeros((len(A), len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                scores[i, j] = np.sum(np.abs(A[i] - sum_norm_B[j]))
        return -scores

    @staticmethod
    def penalty_MAE(A: npt.NDArray, B: npt.NDArray):
        scores = np.zeros((len(A), len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                a, b = A[i], B[j]
                # 某些母离子在库中的峰少于规定的峰数量，因此会进行补全
                # 对于补全的峰，不需要进行惩罚项处理
                non_empty_indices = np.where(a != 0)[0]
                a = a[non_empty_indices]
                b = b[non_empty_indices]
                b = b / np.sum(b)
                b[b == 0] = 1
                scores[i, j] = np.sum(np.abs(a - b))
        return -scores

    @staticmethod
    def peaksum(A: npt.NDArray, B: npt.NDArray):
        max_norm_B = B / np.max(B)
        scores = np.zeros((len(A), len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                scores[i, j] = np.sum(max_norm_B[j])
        return scores
    
    @staticmethod
    def MAE_peaksum(A: npt.NDArray, B: npt.NDArray):
        peaksum_score = FilterSimilarityFunction.peaksum(A, B)
        penalty_MAE_score = FilterSimilarityFunction.MAE(A, B)
        return peaksum_score + penalty_MAE_score

    @staticmethod
    def penalty_MAE_peaksum(A: npt.NDArray, B: npt.NDArray):
        peaksum_score = FilterSimilarityFunction.peaksum(A, B)
        penalty_MAE_score = FilterSimilarityFunction.penalty_MAE(A, B)
        return peaksum_score + penalty_MAE_score

class SpectrumMatcher(ABC):
    def __init__(
        self,
        data_name: str,
        save_dir_path: str,
        library_path: str,
        massspectrum_file_list: List[str],
        identify_labels: Dict[str, Any],
        quant_labels: Dict[str, Any],
        match_params_config: Dict,
        flag_params_config: Dict
    ):
        """
        图谱匹配基类
        """
        self.data_name = data_name
        self.library_path = library_path
        self.massspectrum_file_list = massspectrum_file_list
        self.identify_labels = identify_labels
        self.quant_labels = quant_labels
        self.save_dir_path = save_dir_path
        self.match_params_config = match_params_config
        self.flag_params_config = flag_params_config
        self.create_dirs()

    @property
    def delta(self):
        return self.match_params_config['tol'] * 1e-6
    
    @property
    def match_ms2peak_num_minthreshold(self):
        return self.match_params_config['match_ms2peak_num']
    
    @property
    def peptide_peak_num(self):
        return self.match_params_config['peptide_peak_num']

    @property
    def filter_ms2_num(self):
        return self.match_params_config['filter_ms2_num']

    @property
    def max_process(self):
        return self.match_params_config['num_workers']

    def create_dirs(self):
        start = time.time()
        # 最外层文件夹，子目录为每个文件，再下层为每个文件对应的训练数据以及测试数据
        
        for task_type in ['train', 'test']:
            task_path = os.path.join(self.parent_path, task_type)

            identification_path = os.path.join(task_path, 'identification')
            quantification_path = os.path.join(task_path, 'quantification')
            create_dir(
                identification_path
            )
            create_dir(
                quantification_path
            )
        end = time.time()
        print('所有目录创建完毕 用时 {} 秒'.format(end - start))
    
    @property
    def parent_path(self):
        return os.path.join(self.save_dir_path, self.data_name)

    @property
    def match_ms2_dir(self):
        return os.path.join(self.parent_path, 'match_ms2')

    def match_ms2_filepath(self, file_name: str):
        return os.path.join(self.match_ms2_dir, file_name, 'collection.npy')
    
    def filter_ms2_filepath(self, filter_func_name: Literal['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'penalty_MAE_peaksum', 'MAE_peaksum'], file_name: str):
        return os.path.join(self.filter_ms2_dir, filter_func_name, file_name, 'collection.npy')

    def get_filter_ms2_filename(self, match_ms2_filepath):
        dir, _ = os.path.split(match_ms2_filepath)
        _, file_name = os.path.split(dir)
        return file_name

    @property
    def filter_ms2_dir(self):
        return os.path.join(self.parent_path, 'filter_ms2')
    
    @property
    def train_data_dir(self):
        return os.path.join(self.parent_path, 'train')

    @property
    def test_data_dir(self):
        return os.path.join(self.parent_path, 'test')
    
    @property
    def train_identification_dir(self):
        return os.path.join(self.train_data_dir, 'identification')

    @property
    def test_identification_dir(self):
        return os.path.join(self.test_data_dir, 'identification')
    
    @property
    def train_quantification_dir(self):
        return os.path.join(self.train_data_dir, 'quantification')

    @property
    def test_quantification_dir(self):
        return os.path.join(self.test_data_dir, 'quantification')

    def __multi_process(self, files, func):
        start = time.time()
        with ProcessPoolExecutor(max_workers=self.max_process) as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    func,
                    file
                )
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
        end = time.time()
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('所有文件处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(hour, minute,  second))
        print('-'*120)

    def window_match(
        self, 
        window: Tuple,
        spectrums: Sequence,
        library: Dict,
    ):
        delta = self.delta
        peptide_match_info = {}
        # 字典可能是随机读的，记录读取顺序，以便回读 (不过 python 从哪个版本之后字典都是按顺序读取了，不过还是要向下兼容)
        modifiedcharge_key = []
        peptide_mzs = []
        for modified_charge, metadata in library.items(): # 提取处于该窗口内的肽段信息
            peptide_match_info[modified_charge] = self.init_peptide_info(metadata, window)
            modifiedcharge_key.append(modified_charge)
            peptide_mzs.append(metadata['Spectrum'][:, 0])
        
        modifiedcharge_key = np.array(modifiedcharge_key, dtype=object)
        peptide_mzs = np.array(peptide_mzs)

        # for spectrum in tqdm(spectrums, total=len(spectrums)):
        for spectrum in spectrums:
            mz, intensity = spectrum[0][:, 0], spectrum[0][:, 1] 
            transverse_mz = np.sort(
                np.concatenate(
                    (mz * (1 - delta), mz * (1 + delta)), 
                    axis=0
                )
            )
            insert_index = search(transverse_mz, peptide_mzs)
            match_peak_num = np.sum(insert_index % 2 == 1, axis=1)
            index_ge_threshould = np.nonzero(match_peak_num >= self.match_ms2peak_num_minthreshold)
            matched_peak_pos = insert_index[index_ge_threshould]
            critical_precursor = modifiedcharge_key[index_ge_threshould]
            for pos, modified_charge in zip(matched_peak_pos, critical_precursor):
                peaks = np.zeros((self.peptide_peak_num, 2))
                flag = (pos % 2 == 1)
                peak_pos = (pos[flag] - 1) // 2 # 在实验图谱中的下标位置
                peaks[flag] = np.column_stack((mz[peak_pos], intensity[peak_pos]))
                self.store_ms2_metadata(peptide_match_info[tuple(modified_charge)]['candidate_ms2_metadata'], peaks, self.get_spectrum_metadata(spectrum))
        
        statisfy_peptide_info = {}    
        
        for key in peptide_match_info.keys():
            if len(peptide_match_info[key]['candidate_ms2_metadata']['peaks']) > 0:
                statisfy_peptide_info[key] = peptide_match_info[key]
                self.ms2_metadata_toarray(statisfy_peptide_info[key]['candidate_ms2_metadata'])

        del spectrums
        del library
        return statisfy_peptide_info

    def file_match(self, msfile_path: str):
        _, file = os.path.split(msfile_path)
        file_name = file.split('.npy')[0]
        print(f'read {file_name} data')
        start = time.time()
        read_start = time.time()
        spectrums: Dict = read_dict_npy(msfile_path)
        read_end = time.time()
        hour, minute,  second = transform_time(read_end - read_start)
        print('end, 消耗 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(hour, minute,  second))
        window_seq = spectrums.keys()
        print('divide library by windows!')
        library = divideLibraryByWindows(
            self.library_path,
            window_seq,
            self.match_params_config['tol'],
            self.flag_params_config['is_featuredIons']
        )
        print('end')
        file_matched_ms_info = {}
        for window in window_seq:
            if len(library[window]) == 0:
                print(f"there isn't any peptide of the library in that window {window}")
                continue
            data = self.window_match(
                window,
                spectrums[window],
                library[window],
            )
            file_matched_ms_info.update(data)

        file_dir = os.path.join(self.match_ms2_dir, file_name)
        end = time.time()
        create_dir(file_dir)
        np.save(os.path.join(file_dir, 'collection.npy'), file_matched_ms_info)
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('{} 文件处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(file_name, hour, minute,  second))
        print('-'*120)
        del spectrums
        del library

    def __multi_process_match(self):
        self.__multi_process(self.massspectrum_file_list, self.file_match)
    
    def match(self):
        self.__multi_process_match()

    @abstractmethod
    def cover_matched_metadata(self, candidate_ms2_metadata: Dict, indices: npt.NDArray):
        pass

    @property
    def match_ms2_filepath_list(self):
        match_ms2_filepath_list = []
        for f in self.massspectrum_file_list:
            _, file = os.path.split(f)
            file_name = file.split('.')[0]
            match_ms2_filepath_list.append(self.match_ms2_filepath(file_name))
        return match_ms2_filepath_list

    def file_filter(self, match_ms2_filepath: str):
        file_name = self.get_filter_ms2_filename(match_ms2_filepath)
        start = time.time()
        for filter_func_name in ['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'penalty_MAE_peaksum', 'MAE_peaksum']:
            matched_data: Dict[str, Dict] = np.load(match_ms2_filepath, allow_pickle=True).item()
            filter_func: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = getattr(FilterSimilarityFunction, filter_func_name)
            for metadata in matched_data.values():
                spectrum = metadata['Spectrum']
                peaks = metadata['candidate_ms2_metadata']['peaks']
                ref_intensity = spectrum[:, 1]
                ex_intensity = peaks[:, :, 1]
                scores = filter_func([ref_intensity], ex_intensity)[0]
                argmax_indices = np.argsort(scores)[::-1][:self.filter_ms2_num]
                argmax_indices = np.sort(argmax_indices)
                self.cover_matched_metadata(metadata['candidate_ms2_metadata'], argmax_indices)
                metadata['filter_ms2_metadata'] = metadata.pop('candidate_ms2_metadata')
            save_path = self.filter_ms2_filepath(filter_func_name, file_name)
            dir, _ = os.path.split(save_path)
            create_dir(dir)
            np.save(save_path, matched_data)
        end = time.time()
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('{} 文件处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(file_name, hour, minute,  second))
        print('-'*120)

    def __multi_process_filter(self):
        match_ms2_filepath_list = self.match_ms2_filepath_list
        
        self.__multi_process(match_ms2_filepath_list, self.file_filter)

        # for f in match_ms2_filepath_list:
        #     self.file_filter(f)

    def filter(self):
        self.__multi_process_filter()

    def pipeline(self):
        # self.match()
        self.filter()
        self.generate_train_test_data()

    @abstractmethod
    def _init_identification_label(self):
        pass

    @abstractmethod
    def _init_quantification_label(self):
        pass
    
    @property
    @abstractmethod
    def identification_dtype(self):
        pass

    @property
    @abstractmethod
    def quantification_dtype(self):
        pass

    @abstractmethod
    def identification_item(self, metadata, file_name, key, label):
        pass

    @abstractmethod
    def quantification_item(self, metadata, file_name, key, label):
        pass

    def __multi_process_label(self, files: List[str], task_type: Literal['identification', 'quantification'], filter_type: Literal['MAE', 'penalty_MAE', 'peaksum', 'penalty_MAE_peaksum']):
        
        """
        **********************************
        **千万别用 np.append, 这个操作巨慢**
        **********************************
        
        先用 List 装载 array，最后整合的时候用 np.concatenate(data, axis=0), 如果每个 item 维度一致也可以使用 np.array(data)

        最好还是使用 np.concatenate, 最稳妥
        """
        train_data = []
        test_data = []

        if task_type == 'identification':
            train_save_path = os.path.join(self.train_identification_dir, filter_type, 'collection.npy')
            test_save_path = os.path.join(self.test_identification_dir, filter_type, 'collection.npy')
            func = self.identification_train_test_data
        else:
            train_save_path = os.path.join(self.train_quantification_dir, filter_type, 'collection.npy')
            test_save_path = os.path.join(self.test_quantification_dir, filter_type, 'collection.npy')
            func = self.quantification_train_test_data

        start = time.time()
        with ProcessPoolExecutor(max_workers=self.max_process) as executor:
            futures = []
            for file in files:
                future = executor.submit(
                    func,
                    file,
                    filter_type
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    # 处理结果
                    train, test = future.result()
                    if len(train) > 0:
                        train_data.append(train)
                    test_data.append(test)
                except Exception as e:
                    print(f"任务执行时发生错误: {e}")
                finally:
                    # 显式删除 future 并触发垃圾回收
                    futures.remove(future)
                    gc.collect()
        train_data = np.concatenate(train_data, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        create_dir(os.path.dirname(train_save_path))
        create_dir(os.path.dirname(test_save_path))
        print('开始存储数据')
        load_start = time.time()
        np.save(train_save_path, train_data)
        np.save(test_save_path, test_data)
        load_end = time.time()
        hour, minute, second = transform_time(load_end - load_start)
        print('存储数据消耗 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(hour, minute,  second))
        end = time.time()
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('所有文件处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(hour, minute,  second))
        print('-'*120)

    @property
    def file_name_seq(self):
        files = []
        for f in self.massspectrum_file_list:
            _, file = os.path.split(f)
            file_name = file.split('.')[0]
            files.append(file_name)
        return files

    def generate_identification_train_test_data(self, files: List[str], filter_type: Literal['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'MAE_peaksum', 'penalty_MAE_peaksum']):
        self.__multi_process_label(files, 'identification', filter_type)

    def generate_quantification_train_test_data(self, files: List[str], filter_type: Literal['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'MAE_peaksum', 'penalty_MAE_peaksum']):
        self.__multi_process_label(files, 'quantification', filter_type)

    def fill_with_zero(self, data: npt.NDArray, target_length: int, shape: Tuple):
        mask = np.zeros((self.filter_ms2_num,), dtype=np.bool_)
        if data.shape[0] >= target_length:
            return data, mask
        
        mask[data.shape[0]:] = True
        return np.row_stack(
            (data, np.zeros((target_length - data.shape[0], *shape)))
        ), mask

    def identification_train_test_data(self, file_name: str, filter_type: Literal['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'MAE_peaksum', 'penalty_MAE_peaksum']):
        start = time.time()
        train_data = []
        test_data = []
        filter_ms2_path = self.filter_ms2_filepath(filter_type, file_name)
        filter_ms2_data = read_dict_npy(filter_ms2_path)

        def process_item(key, metadata):
            label_test = 1
            if metadata['decoy']:
                label_test = 0
            item_test = self.identification_item(metadata, file_name, key, label_test)
            if file_name in self.identify_labels:
                if metadata['decoy']:
                    item_train = self.identification_item(metadata, file_name, key, 0)
                    return item_train, item_test
                elif key in self.identify_labels[file_name]:
                    item_train = self.identification_item(metadata, file_name, key, 1)
                    return item_train, item_test
            return None, item_test

        # for key, metadata in filter_ms2_data.items():
        #     process_item(key, metadata)

        with ThreadPoolExecutor() as executor:
            futures = []
            for k, v in filter_ms2_data.items():
                future = executor.submit(process_item, k, v)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    item_train, item_test = future.result()
                    if item_train is not None:
                        train_data.append(item_train)
                    test_data.append(item_test)
                except Exception as e:
                    print(f"任务执行时发生错误: {e}")
                finally:
                    # 显式删除 future
                    futures.remove(future)
        
        if len(train_data) > 0:
            train_data = np.concatenate(train_data, axis=0)
        test_data = np.concatenate(test_data, axis=0)

        print(f'训练数据个数: {len(train_data)}\n测试数据个数: {len(test_data)}')

        end = time.time()
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('{} 处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(file_name, hour, minute,  second))
        print('-'*120)
        return train_data, test_data

    def calculate_xic_correlation(self, ex_ms2, mask):
        xic = ex_ms2[~mask, :, 1].T
        analysis_indices = np.any(xic > 0, axis=1).nonzero()[0]
        empty_indices = np.all(xic == 0, axis=1).nonzero()[0]
        xic = xic[analysis_indices]
            
        if xic.shape[1] == 1:
            return np.zeros(ex_ms2.shape[1])

        coeff = np.corrcoef(xic)
        index = np.sum(coeff, axis=1).argmax()
        corr_vec = coeff[index]
        for index in empty_indices:
            corr_vec = np.insert(corr_vec, index, 0)
        
        return corr_vec

    def ppm(self, ex_ms2, spectrum):
        exms2_mz = ex_ms2[:, :, 0]
        s_mz = spectrum[:, 0]
        
        mz = np.full_like(exms2_mz, -1)
        empty_indices = (exms2_mz == 0).nonzero()
        non_empty_indices = (exms2_mz > 0).nonzero()

        mz[non_empty_indices] = exms2_mz[non_empty_indices]

        ppm_values = np.abs((s_mz - mz) / (mz))

        ppm_values = ppm_values / 1e-6

        ppm_values[empty_indices] = -1

        return ppm_values

    def quantification_train_test_data(self, file_name: str, filter_type: Literal['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'MAE_peaksum', 'penalty_MAE_peaksum']):
        start = time.time()
        train_data = []
        test_data = []
        filter_ms2_path = self.filter_ms2_filepath(filter_type, file_name)
        filter_ms2_data = read_dict_npy(filter_ms2_path)

        def process_item(key, metadata):
            if not metadata['decoy']:
                item_test = self.quantification_item(metadata, file_name, key, 0)
                if file_name in self.quant_labels and key in self.identify_labels[file_name]:
                    item_train = self.quantification_item(metadata, file_name, key, self.identify_labels[file_name][key])
                    return item_train, item_test
                return None, item_test

        with ThreadPoolExecutor() as executor:
            futures = []
            for k, v in filter_ms2_data.items():
                future = executor.submit(process_item, k, v)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    item_train, item_test = future.result()
                    if item_train is not None:
                        train_data.append(item_train)
                    test_data.append(item_test)
                except Exception as e:
                    print(f"任务执行时发生错误: {e}")
                finally:
                    # 显式删除 future
                    futures.remove(future)
        
        if len(train_data) > 0:
            train_data = np.concatenate(train_data, axis=0)
        test_data = np.concatenate(test_data, axis=0)

        print(f'训练数据个数: {len(train_data)}\n测试数据个数: {len(test_data)}')

        end = time.time()
        hour, minute, second = transform_time(end - start)
        print('-'*120)
        print('{} 处理完毕，用时 {:.2f} 时 {:.2f} 分 {:.2f} 秒'.format(file_name, hour, minute,  second))
        print('-'*120)
        return train_data, test_data   

    def generate_train_test_data(self):
        files = self.file_name_seq
        for filter_type in ['cosine', 'MAE', 'penalty_MAE', 'peaksum', 'MAE_peaksum', 'penalty_MAE_peaksum']:
            self.generate_identification_train_test_data(files, filter_type)
            # self.generate_quantification_train_test_data(files, 'penalty_MAE_peaksum')

    """
        ----------------------------------------------
                        必须继承重写的函数
        ----------------------------------------------
        由于质谱数据本身存在多样性，因此所需存储的元数据可能不同，
        
        -   例如普通的质谱数据可能只需要存储峰、保留时间和标号
        -   而 .d 数据则需要存储峰、保留时间、标号还有淌度信息
        
        因此，为了保持这种多样性，需要重写下方的函数       
    """
    
    @abstractmethod
    def store_ms2_metadata(self):
        pass
    
    @abstractmethod
    def get_spectrum_metadata(self):
        pass

    @abstractmethod
    def init_peptide_info(self, metadata, window):
        return {
            # meta data of the modified_peptide
            'decoy': metadata['decoy'],
            'ProteinGroups': metadata['ProteinGroups'] if 'ProteinGroups' in metadata else '',
            'StrippedPeptide': metadata['StrippedPeptide'] if 'StrippedPeptide' in metadata else '',
            'Spectrum': metadata['Spectrum'] ,
            'Fragment': metadata['Fragment'] if 'Fragment' in metadata else '',
            'FeaturedIons': metadata['FeaturedIons'] if 'FeaturedIons' in metadata else [],
            'window': window,
            # matched experimental ms2 info
            'candidate_ms2_metadata': {
                'peaks': [],
                'rt': [],
                'id': []
            }
        }
    
    @abstractmethod
    def ms2_metadata_toarray(self):
        pass

class TimsMatcher(SpectrumMatcher):
    def __init__(
        self, 
        data_name: str, 
        save_dir_path: str, 
        library_path: str, 
        massspectrum_file_list: List[str], 
        identify_labels: Dict[str, Any], 
        quant_labels: Dict[str, Any], 
        match_params_config: Dict, 
        flag_params_config: Dict
    ):
        super().__init__(data_name, save_dir_path, library_path, massspectrum_file_list, identify_labels, quant_labels, match_params_config, flag_params_config)
    
    @property
    def RT_located_index(self):
        return 2

    @property
    def frameid_located_index(self):
        return 3
    
    @property
    def scanid_located_index(self):
        return 4
    
    @property
    def ionmobility_located_index(self):
        return -1
    
    def store_ms2_metadata(
        self,
        candidate_ms2_metadata: Dict[str, List],
        peaks: npt.NDArray,
        metadata
    ):
        candidate_ms2_metadata['peaks'].append(peaks)
        candidate_ms2_metadata['rt'].append(metadata['rt'])
        candidate_ms2_metadata['id'].append(metadata['id'])
        candidate_ms2_metadata['ionmobility'].append(metadata['ionmobility'])
    
    def get_spectrum_metadata(self, spectrum):
        return {
            'rt': spectrum[self.RT_located_index],
            'id': (spectrum[self.frameid_located_index], spectrum[self.scanid_located_index]),
            'ionmobility': spectrum[self.ionmobility_located_index]
        }

    def init_peptide_info(self, metadata, window):
        info = super().init_peptide_info(metadata, window)
        info['ionmobility'] = metadata['IonMobility']
        info['candidate_ms2_metadata']['ionmobility'] = []
        return info
    
    def window_match(self, window: Tuple, spectrums: Sequence, library: Dict):
        start = time.time()
        data = super().window_match(window, spectrums, library)
        end = time.time()
        print('window match consume {:.3f} 秒'.format(end - start))
        return data

    def ms2_metadata_toarray(self, candidate_ms2_metadata):
        for key in ['peaks', 'rt', 'id', 'ionmobility']:
            candidate_ms2_metadata[key] = np.array(candidate_ms2_metadata[key])

    def cover_matched_metadata(self, candidate_ms2_metadata: Dict, indices: npt.NDArray):
        for key in ['peaks', 'rt', 'id', 'ionmobility']:
            candidate_ms2_metadata[key] = candidate_ms2_metadata[key][indices]

class MzmlMatcher(SpectrumMatcher):
    def __init__(
        self, 
        data_name: str, 
        save_dir_path: str, 
        library_path: str, 
        massspectrum_file_list: List[str], 
        identify_labels: Dict[str, Any], 
        quant_labels: Dict[str, Any], 
        match_params_config: Dict, 
        flag_params_config: Dict
    ):
        super().__init__(data_name, save_dir_path, library_path, massspectrum_file_list, identify_labels, quant_labels, match_params_config, flag_params_config)

    @property
    def identification_dtype(self):
        return np.dtype(
            [
                ('Ex_ms2', np.float32, (self.filter_ms2_num, self.peptide_peak_num, 2)), 
                ('Spectrum', np.float32, (1, self.peptide_peak_num, 2)), 
                ('Featured_ion', np.float32, (self.peptide_peak_num,)), 
                ('Mask', np.bool_, (self.filter_ms2_num,)), 
                ('Xic_corr', np.float32, (self.peptide_peak_num,)),
                ('Ppm', np.float32, (self.filter_ms2_num, self.peptide_peak_num)),
                ('Label', np.float32), 
                ('Info', object)
            ])

    @property
    def quantification_dtype(self):
        pass

    @property
    def RT_located_index(self):
        return 2
    
    @property
    def id_loacated_index(self):
        return -1

    def _init_identification_label(self):
        return np.array([], dtype=self.identification_dtype)
    
    def _init_quantification_label(self):
        return np.array([], dtype=self.quantification_dtype)

    def identification_item(self, metadata, file_name, key, label):

        ex_ms2, mask = self.fill_with_zero(
                metadata['filter_ms2_metadata']['peaks'],
                self.filter_ms2_num,
                (self.peptide_peak_num, 2)
            )
        
        spectrum = metadata['Spectrum']
        xic_corr = self.calculate_xic_correlation(ex_ms2, mask)
        ppm_ = self.ppm(ex_ms2, spectrum)
        featured_ions = metadata['FeaturedIons']
        info = (file_name, metadata['window'], key)

        return np.array(
            [(
                ex_ms2,
                spectrum,
                featured_ions,
                mask,
                xic_corr,
                ppm_,
                label,
                info
            )],
            dtype=self.identification_dtype
        )

    def quantification_item(self):
        pass

    def store_ms2_metadata(
        self,
        candidate_ms2_metadata: Dict[str, List],
        peaks: npt.NDArray,
        metadata: Dict
    ):
        candidate_ms2_metadata['peaks'].append(peaks)
        candidate_ms2_metadata['rt'].append(metadata['rt'])
        candidate_ms2_metadata['id'].append(metadata['id'])

    def ms2_metadata_toarray(self, candidate_ms2_metadata):
        for key in ['peaks', 'rt', 'id']:
            candidate_ms2_metadata[key] = np.array(candidate_ms2_metadata[key])

    def init_peptide_info(self, metadata, window):
        return super().init_peptide_info(metadata, window)

    def get_spectrum_metadata(self, spectrum):
        return {
            'id': spectrum[self.id_loacated_index],
            'rt': spectrum[self.RT_located_index]
        }
    
    def window_match(self, window: Tuple, spectrums: Sequence, library: Dict):
        start = time.time()
        data = super().window_match(window, spectrums, library)
        end = time.time()
        print('window match consume {:.3f} 秒'.format(end - start))
        return data

    def cover_matched_metadata(self, candidate_ms2_metadata: Dict, indices: npt.NDArray):
        for key in ['peaks', 'id', 'rt']:
            candidate_ms2_metadata[key] = candidate_ms2_metadata[key][indices]