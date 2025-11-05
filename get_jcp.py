from utils.model_utils import *
from utils.utils import read_mks_data
import pandas as pd
# === Load data ===
df = pd.read_csv("/home/kchalabi/Documents/THESE/datasets_kinetics/Human_data/Vincent/Trial109_mks.csv")

def as_col(x):
        x = np.asarray(x)
        return x.reshape(3, 1) if x.shape != (3, 1) else x


mks_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)
for i, frame in enumerate(mks_dict):
    print(frame)
    input()
    pelvis_pose = get_pelvis_pose(frame)
    HJC = as_col(pelvis_pose[:3, 3])
    KNE_lat = frame['LKNE']
    TIB =frame ['RTIB']
    RTHI = frame ['RTHI']
    side = 'right'


    knee_jcp = knee_joint_center(
        HJC, KNE_lat, RTHI, 0.5, side,
        knee_rotation_offset_deg=0.0, MKNE_med=None, eps=1e-12
    )
    print(knee_jcp)
# KJC = [0.10, 0.50, 0.10] # knee JC (m)
# ANK = [0.12, 0.10, 0.11] # lateral malleolus (m)
# TIB = [0.17, 0.25, 0.12] # shank wand (lateral) (m)
# ankle_width = 0.07 # 7 cm
# ajc = ankle_joint_center(KJC, ANK, TIB, ankle_offset=0.5*ankle_width,
# side="right", ankle_rotation_offset_deg=0.0)
# print("AJC:", ajc)