# https://omoroya.com/arduino-lesson32/
# https://msr-r.net/m5stickc-sdcard-time-interval/
# https://www.ei.tohoku.ac.jp/xkozima/lab/espTutorial2.html
# https://shizenkarasuzon.hatenablog.com/entry/2019/02/16/181342
# https://watako-lab.com/2019/02/28/3axis_gyro/
# https://shizenkarasuzon.hatenablog.com/entry/2019/02/16/170706
# https://ahrs.readthedocs.io/en/latest/
# https://pypi.org/project/AHRS/

#%%
import numpy as np
import os
import sys
import glob
from natsort import natsorted
from scipy.optimize import curve_fit

import math
import sklearn
#%%
# %%
os.path.dirname(__file__)
# %%
folder = "./csv_noise"
#%%
file = folder+"/IMU_calibrated.csv"
data=np.loadtxt(file, delimiter=',')
# %%
data
# %%
