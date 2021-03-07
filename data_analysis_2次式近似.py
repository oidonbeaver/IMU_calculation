#%%
# import function
import numpy as np
import matplotlib.pyplot as plt
import csv
from natsort import natsorted
from scipy.optimize import curve_fit
import math
import pandas as pd
# %%
import os
#%%
os.path.dirname(__file__)
#%%

# %%
folder_name = "/"
# %%
# scan_list=natsorted(os.listdir(os.path.dirname(__file__)+folder_name))
# %%
fig_colors = ["red","blue","fuchsia","darkviolet","darkgreen","gold","black","lime","gray","orange","brown"]
markers=[".",",","v","^","<",">","1","2","3","4","8","p","*","h","H","+","D","x","|","_","s"]
fig_colors=fig_colors*2

scan_list=np.array(["SCAN2"])#全部
# scan_list=["SCAN8","SCAN19","SCAN2","SCAN7","SCAN21","SCAN16","SCAN4","SCAN18","SCAN11","SCAN25","SCAN14","SCAN10","SCAN5","SCAN24","SCAN12","SCAN15","SCAN3","SCAN17","SCAN9","SCAN22","SCAN6"]#周波数ごと
labels=["1rps"]

# scan_ferq_index=np.array([1,0,7,4,3,6,8,2,5])#1rps,5rps,10rps....40rpsの順
scan_ferq_index=np.arange(21)#スピード順にscan_listを並べたので
# %%
scan_list[scan_ferq_index]
#%%
scan_num=5
# %%



enc_start=346
# enc_end=enc_start+5
enc_end=350

# %%
# datalist=[]
for i in range(len(scan_list)):
    scan_num=i

    folder_path = os.path.dirname(__file__)+folder_name+scan_list[scan_num]

    csv_filepath = folder_path+"/ScanDebug.csv"

    # ラベル付きで読み込む、[0,0]のように要素を指定できない
    a= np.genfromtxt(csv_filepath,delimiter=",",skip_header=1,names=True,usecols=[0,2,5,6,7],dtype="uint32")
    # use_bool = (a["Enc2cnt"]<15*65856/360) |  (a["Enc2cnt"]>330*65856/360)
    use_bool1 = (a["Enc2cnt"]<(enc_end)*65856/360) &  (a["Enc2cnt"]>enc_start*65856/360)
    
    Erro_status = a["Erro_Status"]
    use_bool2 = Erro_status == 0
    use_bool = use_bool1 & use_bool2
    
    
    index = np.ones(len(use_bool))[use_bool]*(i)
    # time_ms = (a["Time_Stamp"]-a["Time_Stamp"][0])/1.875/10**3
    distance_m = a["Dist"][use_bool]/2**6/10**3
    enc1_deg = a["Enc1cnt"][use_bool]/65856*360
    enc1_deg[enc1_deg>180] = enc1_deg[enc1_deg>180]-360
    enc2_deg = a["Enc2cnt"][use_bool]/65856*360
    
    # mat=np.array([index,distance_m,enc1_deg,enc2_deg]).T
    mat=np.array([index,distance_m,enc2_deg]).T
    if i == 0:
        data_matrix = mat
    else:
        data_matrix = np.concatenate([data_matrix,mat])
    # datalist.append(mat)
#%%
del mat
del a
# del time_ms
del distance_m
del enc1_deg
del enc2_deg
del use_bool
del use_bool1
del use_bool2
del Erro_status
#%%
def func4(X,a,b,c,d):
    x=X
    return a + b*x**3 + c*x**2 + d*x
def func5(X,a,b,c,d,e):
    x=X
    return a + b*x**4 + c*x**3 + d*x**2 + e*x
# %%
p0 = 1., 1., 1.,1.
x=data_matrix[:,2]
# y=data_matrix[:,2]
z=data_matrix[:,1]
popt, pcov =curve_fit(func4, x, z, p0)
residuals =  z- func4(x, popt[0],popt[1],popt[2],popt[3])
# %%
rss = np.sum(residuals**2)#residual sum of squares = rss
tss = np.sum((z-np.mean(z))**2)#total sum of squares = tss
r_squared = 1 - (rss / tss)
# %%
r_squared#決定係数
#%%
# rmse=rss**0.5
popt
# %%




# %%
# 全データのときは描画しない
rmselist=[]
for i in scan_ferq_index:
    x=data_matrix[:,2][data_matrix[:,0]==i]
    # y=data_matrix[:,2][data_matrix[:,0]==i+1]
    z=data_matrix[:,1][data_matrix[:,0]==i]
    residuals =  z- func4(x, popt[0],popt[1],popt[2],popt[3])
    rmse = np.mean(residuals**2)**0.5
    rmselist.append(rmse)

#%%


# %%
rmselist
# %%
# %%
individual=1
# %%
# fig = plt.figure(figsize=(12,10),dpi=500)
fig = plt.figure(figsize=(12,10),dpi=500)
fig2 = plt.figure(figsize=(12,10),dpi=500)
ax1 = fig.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)
meds = np.array([])
t=0
for i in scan_ferq_index:
    x=data_matrix[:,2][data_matrix[:,0]==i]
    # y=data_matrix[:,2][data_matrix[:,0]==i+1]
    z=data_matrix[:,1][data_matrix[:,0]==i]

  
    p0 = 1., 1., 1.,1.
    popt_ind, pcov_ind =curve_fit(func4, x, z, p0)
    residuals =  z- func4(x, popt_ind[0],popt_ind[1],popt_ind[2],popt_ind[3])
    if t==0:
        residuals_list = residuals
    else: 
        residuals_list = np.append(residuals_list,residuals)

 
    
    ax1.set_xticks( np.arange(enc_start,enc_end+1,1) )
    label = labels[i] +", "+ scan_list[i] + ", " + str(len(x)) +"points"
    ax1.scatter(x,residuals,s=0.8,color=fig_colors[i],label=label,marker=markers[i])
    
    ax2.scatter(x,z,s=0.8,color=fig_colors[i],label=label,marker=markers[i])
    meds = np.append(meds,math.floor(np.nan_to_num(np.median(z))*10)/10)#点が無いとき(nanのとき)　ゼロにする
    t +=1

ax1.set_xticks( np.arange(enc_start,enc_end+1,1) )
ax1.set_xlabel("Enc2の角度(deg)", fontname="MS Gothic")
ax1.set_ylabel("測距の残差(m)", fontname="MS Gothic")    
    # ax1.savefig("img.png")

ax1.set_ylim(-0.03,0.03) 
ax1.legend(markerscale=8)

ax2.set_xticks( np.arange(enc_start,enc_end+1,1) )
ax2.set_xlabel("Enc2の角度(deg)", fontname="MS Gothic")
ax2.set_ylabel("測距値(m)", fontname="MS Gothic")
ax2.legend(markerscale=8)
meds = meds[meds!=0]
if np.size(meds) == 0:
    med=0
else:
    med= math.floor(np.mean(meds)*10)/10
ax2.set_ylim(med-0.3,med+0.7) 



savefilename1 ="測距残差_"+str(enc_start) + "度から" + str(enc_end)+ "度.png"
fig.savefig(savefilename1)
savefilename2 = "測距値_"+str(enc_start) + "度から" + str(enc_end)+ "度.png"
fig2.savefig(savefilename2)



# %%
data_matrix=np.insert(data_matrix,3,residuals_list,axis=1)
# %%
data_matrix.shape


# %%
df = pd.DataFrame({"id":data_matrix[:,0],"residuals":data_matrix[:,3]})
# %%
df.groupby("id").std()
# %%
data_matrix[:,0]
# %%
