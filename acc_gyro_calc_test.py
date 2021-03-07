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
file = folder+"/"
np.loadtxt(file, delimiter=',')