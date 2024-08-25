import sys
sys.path.append('../')
from read_raw.read_tims_data import main
from utils.transform import cal_time

if __name__ == "__main__":
    @cal_time("所有数据处理完毕")
    def excute():
        for root_path in [
            # '/data/xp/data/1. T2D-CAD-RJ',
            # '/data/xp/data/2. MeS_YN',
            # '/data/xp/data/3. brain cohort-JW',
            '/data/xp/data/DLUP/dataset7-PXD040205'
        ]:
            main(root_path, 30, 15)
    excute()