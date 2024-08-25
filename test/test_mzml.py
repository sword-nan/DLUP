import sys
sys.path.append('../')
from read_raw.read_mzml_data import main
from utils.transform import cal_time

if __name__ == "__main__":
    @cal_time("所有数据处理完毕")
    def excute():
        for root_path in [
            # '/data/xp/data/astral_20231016_300ngPlasmaSample',
            # '/data/xp/data/lfq32',
            '/data/xp/data/NoRT_LFQ',
            '/data/xp/data/Hela'
        ]:
            main(root_path, 15, 15)
    excute()