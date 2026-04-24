from scipy.interpolate import RBFInterpolator
from pickle import load
import numpy as np
import os

disp_path = "/home/teseo/scratch/cell/final_disps.pkl"
centers_path = "/home/teseo/scratch/cell/centers.pkl"

with open(disp_path, "rb") as f:
    disps = load(f)

with open(centers_path, "rb") as f:
    centers = load(f)

#index = 100
index = int(os.environ.get("JOB_INDEX", 0))

rbf_interp = RBFInterpolator(centers, disps[index, :, :])

def rbf(x, y, z):
    return rbf_interp(np.array([[x, y, z]]))

def rbf_x(x, y, z, t, index):
    return rbf(x, y, z)[0, 0]


def rbf_y(x, y, z, t, index):
    return rbf(x, y, z)[0, 1]

def rbf_z(x, y, z, t, index):
    return rbf(x, y, z)[0, 2]

if __name__ == "__main__":
    x = 0.5
    y = 0.5
    z = 0.5
    print(rbf_x(x, y, z, 0, 0))
    print(rbf_y(x, y, z, 0, 0))
    print(rbf_z(x, y, z, 0, 0))