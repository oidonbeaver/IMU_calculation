#%%
import numpy as np
import os
import sys
import glob
from natsort import natsorted
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import sklearn
#%%
def concat_csv(folder):
    filenum=len(glob.glob(folder+"/*"))
    for i in range(filenum):
        file = glob.glob(folder+"/*")[i]
        if (i == 0):
            data_all = np.loadtxt(file, delimiter=',', dtype='int32')
        else:
            data = np.loadtxt(file, delimiter=',', dtype='int32')
            data_all = np.concatenate([data_all, data])
    return data_all
# %%
folder_noise = "./csv_noise"
data_concat = concat_csv(folder_noise)
np.savetxt('./csv_noise/noise_test_all.csv', data_concat, delimiter=',', fmt='%d')
#%%
data_cnv = data_concat.astype(np.float32)
data_cnv[:, [2, 3, 4, 9, 10,11]] = data_concat[:, [2, 3, 4, 9,10, 11]]/16384
data_cnv[:, [5, 6, 7, 12, 13, 14]] = data_concat[:, [5, 6, 7, 12, 13, 14]] / 131
data_calib_cnv = data_cnv

#%%
def func2(X,a,b,c,d):
    x = X
    # return (1 - (((x[0] - a) ** 2 + (x[1] - b) ** 2 + (x[2] - c) ** 2)) ** 0.5) ** 2
    return 1-d*((x[0]-a)**2+(x[1]-b)**2+(x[2]-c)**2)**0.5
            
def func3(X,a,b,c):
    x = X
    # return (1 - (((x[0] - a) ** 2 + (x[1] - b) ** 2 + (x[2] - c) ** 2)) ** 0.5) ** 2
    return 1 - ((x[0] - a) ** 2 + (x[1] - b) ** 2 + (x[2] - c) ** 2)
#%%
# p2 = 0., 0., 0.,1
# x = data_cnv[:, [2, 3, 4]].T
# x = data_cnv[:, [9, 10, 11]].T
# z=np.zeros(x.shape[1])
# # %%
# popt2, pcov =curve_fit(func2, x, z, p2)
# # %%
# curve_fit(func2, x, z, p2)
# %%
# IMU1の加速度をキャリブレーション
p3 = 0., 0., 0.
x = data_cnv[:, [2, 3, 4]].T
# x = data_cnv[:, [9, 10, 11]].T
z=np.zeros(x.shape[1])
popt3, pcov =curve_fit(func3, x, z, p3)
data_calib_cnv[:, [2, 3, 4]] = data_calib_cnv[:, [2, 3, 4]] - popt3
#%%
# IMU2の加速度をキャリブレーション
p3 = 0., 0., 0.
# x = data_cnv[:, [2, 3, 4]].T
x = data_cnv[:, [9, 10, 11]].T
z=np.zeros(x.shape[1])
popt3, pcov =curve_fit(func3, x, z, p3)
data_calib_cnv[:, [9, 10, 11]] = data_calib_cnv[:, [9, 10, 11]] - popt3

# # %%
# residuals2 = z - func3(x, popt2[0], popt2[1], popt2[2])
# residuals3 = z - func3(x, popt3[0], popt3[1], popt3[2])
# print(np.mean(residuals2**2))
# print(np.mean(residuals3**2))
# %%
#%%
# 角速度をキャリブレーション
for k in [5, 6, 7, 12, 13, 14]:
    data_calib_cnv[:,k]=np.median(data_cnv[:, k])
# %%
fmtlist=['%d','%d','% f','% f','% f','% f','% f','% f','%d','% f','% f','% f','% f','% f','% f','%d']
np.savetxt('./csv_noise/IMU_calibrated.csv', data_calib_cnv, delimiter=',', fmt=fmtlist)
# %%

# %%
