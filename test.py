import os

import numpy as np

from spectra_match.matcher import MzmlMatcher, TimsMatcher


def CAD():
    dir = '/data/xp/data/1. T2D-CAD-RJ/merge'

    files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.npy')]

    matcher = TimsMatcher(
        data_name='1. T2D-CAD-RJ',
        save_dir_path='/data/xp/train_test_data',
        library_path='/data/xp/library/AD8_MeS_MN_AC_Brain_AcuteExercise_20240428_Top6_Target_DecoyPsps23_SumNorm.npy',
        massspectrum_file_list=files,
        identify_labels={},
        quant_labels={},
        match_params_config={
            'num_workers': 20,
            'peptide_peak_num': 6,
            'tol': 15,
            'match_ms2peak_num': 3
        },
        flag_params_config={
            'is_featuredIons': True
        }
    )

    matcher.match()

def astral():
    dir = '/data/xp/data/astral_20231016_300ngPlasmaSample/merge'
    label_path = '/data/xp/label/astral_20231016_300ngPlasmaSample/train_identification.npy'

    identification_label = np.load(label_path, allow_pickle=True).item()

    files = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.npy')]
    # files = files[:5]

    matcher = MzmlMatcher(
        data_name='astral_20231016_300ngPlasmaSample',
        save_dir_path='/data/xp/train_test_data',
        library_path='/data/xp/library/AD8_Trypsin_Astral_directDIALibrary_Top6_Target_DecoyPsps23_SumNorm.npy',
        massspectrum_file_list=files,
        identify_labels=identification_label,
        quant_labels={},
        match_params_config={
            'num_workers': 20,
            'filter_ms2_num': 6,
            'peptide_peak_num': 6,
            'tol': 15,
            'match_ms2peak_num': 3
        },
        flag_params_config={
            'is_featuredIons': True
        }
    )
    # files = matcher.match_ms2_filepath_list
    # matcher.file_filter(files[0])
    matcher.pipeline()
    # matcher.match()
    # matcher.filter()
    # files = matcher.file_name_seq
    # print(len(files))
    # matcher.generate_identification_train_test_data(files, 'penalty_MAE_peaksum')
    # for file in files[:1]:
    #     matcher.identification_train_test_data(file, 'penalty_MAE_peaksum')
    # matcher.generate_train_test_data()

if __name__ == "__main__":
    astral()
    # CAD()