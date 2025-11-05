import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import time
import meshcat
from utils import read_mks_data
from viz_utils import add_sphere, place
from model_utils import *

# === Load data ===
df = pd.read_csv("/home/kchalabi/Documents/THESE/datasets_kinetics/Human_data/Vincent/Trial109_mks.csv")


mks_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)

mks_names = start_sample_dict.keys()
# === Initialize Meshcat Visualizer ===
vis = meshcat.Visualizer().open()


for name in mks_names:
    add_sphere(vis, f"world/{name}", radius=0.01, color= 0xff0000)


# === Animate frame by frame ===
for i, frame in enumerate(mks_dict):
    for name in mks_names:
        pos = frame[name].reshape(3,)
        place(vis, name, pos)
    
    time.sleep(0.01)

