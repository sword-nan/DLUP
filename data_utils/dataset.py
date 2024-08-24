import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

class MzmlDataset(Dataset):
    def __init__(self, data: npt.NDArray):
        self.length = len(data)
        self.ex_ms2 = data['Ex_ms2']
        self.spectrum = data['Spectrum']
        self.xic_corr = data['Xic_corr']
        self.ppm = data['Ppm']
        self.featured_ions = data['Featured_ion']
        self.mask = data['Mask']
        self.label = data['Label']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.ex_ms2[index], self.spectrum[index], self.mask[index], self.xic_corr[index], self.ppm[index], self.featured_ions[index]), self.label[index]