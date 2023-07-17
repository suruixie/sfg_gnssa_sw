"""
Copyright (c) 2023, Surui Xie
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 




This program is used to solve for the transpoder position with data collected on 2021/07/23, tested with Python 3.9.5 in macOS v11.6.6

Readers should be aware that some of the settings were used to intentionally produce exactly the same figure as Figure 8 in the published paper.

Example run:
    python pos20210723.py -i data/gnssa_data_20210723.txt

Reference: Xie, S., Zumberge, M., Sasagawa, G., and Voytenko, D., 202x. Shallow Water Seafloor Geodesy with Wave Glider-Based GNSS-Acoustic Surveying of a Single Transponder. Submitted to Earth and Space Science

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as dts
from datetime import datetime, date, time, timedelta, timezone
import sys
from scipy import sparse

colors=['red','blue','dodgerblue','gold','lime','magenta','coral','brown','green','cyan','olive','skyblue']
colors_a=['darkgrey','dimgrey']
colors_comb=colors_a+colors

refN0=0    #A priori transponder coordinate in local topocentric system (North), unit: meter
refE0=0    #A priori transponder coordinate in local topocentric system (East), unit: meter
refU0=0    #A priori transponder coordinate in local topocentric system (Up), unit: meter

hmean_acous_c0=1499.3           #Harmonic mean derived from the mooring measurement: https://mooring.ucsd.edu/delmar1/delmar1_18/ 
hmean_acous_c0_woa18=1500.9     #Harmonic mean derived from WOA18: https://www.ncei.noaa.gov/access/world-ocean-atlas-2018/ 
sigma0=0.03                     #standard deviation used for weight calculations, unit: m
twtT_thres_rms_ratio=3          #Threshold used to remove outliers based on the residual rms

fdata = sys.argv[sys.argv.index("-i")+1]

#segments used for position estimate, non-overlapping
Tstr0=[ ["2021-07-23 19:34:30","2021-07-23 19:50:50"], \
        ["2021-07-23 19:50:50","2021-07-23 20:06:30"], \
        ["2021-07-23 20:11:40","2021-07-23 20:33:30"], \
        ["2021-07-23 20:33:30","2021-07-23 20:55:10"]]
Tseg0=np.zeros([len(Tstr0),2])
for k in range(0,len(Tstr0)):
    Tseg0[k,0]=dts.date2num(datetime.strptime(Tstr0[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg0[k,1]=dts.date2num(datetime.strptime(Tstr0[k][1],"%Y-%m-%d %H:%M:%S"))

#Segment with bad geometry, for test purpose
Tstr0_a = [ ["2021-07-23 20:06:30","2021-07-23 20:26:10"], \
             ["2021-07-23 20:48:20","2021-07-23 21:07:00"] ]
Tseg0_a=np.zeros([len(Tstr0_a),2])
for k in range(0,len(Tstr0_a)):
    Tseg0_a[k,0]=dts.date2num(datetime.strptime(Tstr0_a[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg0_a[k,1]=dts.date2num(datetime.strptime(Tstr0_a[k][1],"%Y-%m-%d %H:%M:%S"))


#segments used for position estimate, with some overlapping
Tstr1=[ ["2021-07-23 19:34:30","2021-07-23 19:50:50"], \
        ["2021-07-23 19:38:40","2021-07-23 19:55:00"], \
        ["2021-07-23 19:43:30","2021-07-23 19:59:40"], \
        ["2021-07-23 19:47:10","2021-07-23 20:03:00"], \
        ["2021-07-23 19:50:50","2021-07-23 20:07:00"], \
        ["2021-07-23 19:55:40","2021-07-23 20:10:40"], \
        ["2021-07-23 20:11:40","2021-07-23 20:33:30"], \
        ["2021-07-23 20:17:10","2021-07-23 20:39:00"], \
        ["2021-07-23 20:22:10","2021-07-23 20:44:00"], \
        ["2021-07-23 20:27:10","2021-07-23 20:49:00"], \
        ["2021-07-23 20:33:30","2021-07-23 20:55:10"], \
        ["2021-07-23 20:39:10","2021-07-23 21:01:00"]]
Tseg1=np.zeros([len(Tstr1),2])
for k in range(0,len(Tstr1)):
    Tseg1[k,0]=dts.date2num(datetime.strptime(Tstr1[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg1[k,1]=dts.date2num(datetime.strptime(Tstr1[k][1],"%Y-%m-%d %H:%M:%S"))

#Segment with bad geometry, for test purpose
Tstr1_a = [ ["2021-07-23 20:06:30","2021-07-23 20:26:10"], \
             ["2021-07-23 20:48:20","2021-07-23 21:07:00"] ]
Tseg1_a=np.zeros([len(Tstr1_a),2])
for k in range(0,len(Tstr1_a)):
    Tseg1_a[k,0]=dts.date2num(datetime.strptime(Tstr1_a[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg1_a[k,1]=dts.date2num(datetime.strptime(Tstr1_a[k][1],"%Y-%m-%d %H:%M:%S"))

UTC_ref = datetime(2020,1,1).replace(tzinfo=timezone.utc)   #second relative to this time


def getAWl(neu0,tneusss0neusss1,apri_sigma,soundv):
    nobs=tneusss0neusss1.shape[0]
    GF=np.empty(shape=[nobs,3])
    Obs=np.empty(shape=[nobs,1])
    Obs_sigma2=np.empty(shape=nobs)
    for i in range(0,nobs):
        r1=np.sqrt( (tneusss0neusss1[i,1]-neu0[0,0])**2 + (tneusss0neusss1[i,2]-neu0[1,0])**2 + (tneusss0neusss1[i,3]-neu0[2,0])**2 )
        r2=np.sqrt( (tneusss0neusss1[i,7]-neu0[0,0])**2 + (tneusss0neusss1[i,8]-neu0[1,0])**2 + (tneusss0neusss1[i,9]-neu0[2,0])**2 )
        GF[i,0] = (neu0[0,0]-tneusss0neusss1[i,1])/r1 + (neu0[0,0]-tneusss0neusss1[i,7])/r2
        GF[i,1] = (neu0[1,0]-tneusss0neusss1[i,2])/r1 + (neu0[1,0]-tneusss0neusss1[i,8])/r2
        GF[i,2] = (neu0[2,0]-tneusss0neusss1[i,3])/r1 + (neu0[2,0]-tneusss0neusss1[i,9])/r2
        Obs[i,0] = soundv*tneusss0neusss1[i,0]-r1-r2
        Obs_sigma2[i] = ((neu0[0,0]-tneusss0neusss1[i,1])*tneusss0neusss1[i,4]/r1)**2  + \
                        ((neu0[1,0]-tneusss0neusss1[i,2])*tneusss0neusss1[i,5]/r1)**2  + \
                        ((neu0[2,0]-tneusss0neusss1[i,3])*tneusss0neusss1[i,6]/r1)**2  + \
                        ((neu0[0,0]-tneusss0neusss1[i,7])*tneusss0neusss1[i,10]/r2)**2  + \
                        ((neu0[1,0]-tneusss0neusss1[i,8])*tneusss0neusss1[i,11]/r2)**2  + \
                        ((neu0[2,0]-tneusss0neusss1[i,9])*tneusss0neusss1[i,12]/r2)**2
    Weights=sparse.spdiags(apri_sigma**2/Obs_sigma2,0,tneusss0neusss1.shape[0],tneusss0neusss1.shape[0])
    return GF, Weights, Obs
    
def getAWl_freeC(neuc0,tneusss0neusss1):
    nobs=tneusss0neusss1.shape[0]
    GF=np.empty(shape=[nobs,4])
    Obs=np.empty(shape=[nobs,1])
    Obs_sigma2=np.empty(shape=nobs)
    for i in range(0,nobs):
        r1=np.sqrt( (tneusss0neusss1[i,1]-neuc0[0,0])**2 + (tneusss0neusss1[i,2]-neuc0[1,0])**2 + (tneusss0neusss1[i,3]-neuc0[2,0])**2 )
        r2=np.sqrt( (tneusss0neusss1[i,7]-neuc0[0,0])**2 + (tneusss0neusss1[i,8]-neuc0[1,0])**2 + (tneusss0neusss1[i,9]-neuc0[2,0])**2 )
        GF[i,0] = (neuc0[0,0]-tneusss0neusss1[i,1])/r1 + (neuc0[0,0]-tneusss0neusss1[i,7])/r2
        GF[i,1] = (neuc0[1,0]-tneusss0neusss1[i,2])/r1 + (neuc0[1,0]-tneusss0neusss1[i,8])/r2
        GF[i,2] = (neuc0[2,0]-tneusss0neusss1[i,3])/r1 + (neuc0[2,0]-tneusss0neusss1[i,9])/r2
        GF[i,3] = -tneusss0neusss1[i,0]
        Obs[i,0] = neuc0[3,0]*tneusss0neusss1[i,0]-r1-r2
        Obs_sigma2[i] = ((neuc0[0,0]-tneusss0neusss1[i,1])*tneusss0neusss1[i,4]/r1)**2  + \
                        ((neuc0[1,0]-tneusss0neusss1[i,2])*tneusss0neusss1[i,5]/r1)**2  + \
                        ((neuc0[2,0]-tneusss0neusss1[i,3])*tneusss0neusss1[i,6]/r1)**2  + \
                        ((neuc0[0,0]-tneusss0neusss1[i,7])*tneusss0neusss1[i,10]/r2)**2  + \
                        ((neuc0[1,0]-tneusss0neusss1[i,8])*tneusss0neusss1[i,11]/r2)**2  + \
                        ((neuc0[2,0]-tneusss0neusss1[i,9])*tneusss0neusss1[i,12]/r2)**2
    Weights=sparse.spdiags(1/Obs_sigma2,0,tneusss0neusss1.shape[0],tneusss0neusss1.shape[0])
    return GF, Weights, Obs

def vecNvecE2AZ(VecN,VecE):  #calculate azimuth angle from north/east vectors
    if ( (VecN>0) & (VecE>=0) ):
        AZ_degree=np.arctan(VecE/VecN)*180.0/np.pi
    elif ( (VecN==0) & (VecE>=0) ):
        AZ_degree=90.0
    elif ( (VecN<0) & (VecE>=0) ):
        AZ_degree=180.0+np.arctan(VecE/VecN)*180.0/np.pi
    elif ( (VecN<0) & (VecE<0) ):
        AZ_degree=180.0+np.arctan(VecE/VecN)*180.0/np.pi
    elif ( (VecN==0) & (VecE<0) ):
        AZ_degree=270.0
    else:
        AZ_degree=360.0+np.arctan(VecE/VecN)*180.0/np.pi
    return AZ_degree




#Initiate the figure
fig = plt.figure(figsize=(6.5,9))
gs = gridspec.GridSpec(9, 5, height_ratios=[2.7,0.35,1,0.,2,0.1,1,1,1], width_ratios=[1,1,1,1,1])
gs.update(hspace=0.17,wspace=0.07)

ax00 = fig.add_subplot(gs[0,0:2])
ax01 = fig.add_subplot(gs[0,3:5])
ax10=fig.add_subplot(gs[2,1:4])
ax20 = fig.add_subplot(gs[4,0])
ax21 = fig.add_subplot(gs[4,1], sharey=ax20)
ax22 = fig.add_subplot(gs[4,2], sharey=ax20)
ax23 = fig.add_subplot(gs[4,3], sharey=ax20)
ax24 = fig.add_subplot(gs[4,4], sharey=ax20)


ax30 = fig.add_subplot(gs[6,0])
ax31 = fig.add_subplot(gs[6,1], sharey=ax30)
ax32 = fig.add_subplot(gs[6,2], sharey=ax30)
ax33 = fig.add_subplot(gs[6,3], sharey=ax30)
ax34 = fig.add_subplot(gs[6,4], sharey=ax30)

ax40 = fig.add_subplot(gs[7,0], sharex=ax30)
ax41 = fig.add_subplot(gs[7,1], sharex=ax31, sharey=ax40)
ax42 = fig.add_subplot(gs[7,2], sharex=ax32, sharey=ax40)
ax43 = fig.add_subplot(gs[7,3], sharex=ax33, sharey=ax40)
ax44 = fig.add_subplot(gs[7,4], sharex=ax34, sharey=ax40)

ax50 = fig.add_subplot(gs[8,0], sharex=ax30)
ax51 = fig.add_subplot(gs[8,1], sharex=ax31, sharey=ax50)
ax52 = fig.add_subplot(gs[8,2], sharex=ax32, sharey=ax50)
ax53 = fig.add_subplot(gs[8,3], sharex=ax33, sharey=ax50)
ax54 = fig.add_subplot(gs[8,4], sharex=ax34, sharey=ax50)

fig.subplots_adjust(left=0.06, bottom=0.043, right=0.975, top=0.99)
Tleft=dts.date2num(datetime.strptime(str(202107231927),"%Y%m%d%H%M"))
Tright=dts.date2num(datetime.strptime(str(202107232108),"%Y%m%d%H%M"))


for ax in [ax00,ax01]:
    ax.set_aspect('equal')
    ax.set_xlim(-168,168)
    ax.set_ylim(-168,168)
    ax.set_xticks([-100,0,100])
    ax.set_yticks([-100,0,100])
    ax.set_xlabel("East (m)",labelpad=0)
    ax.set_ylabel("North (m)",labelpad=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

ax10.set_xlim(Tleft,Tright)
ax10.set_ylim(0,0.32)
ax10.set_yticks([0,0.1,0.2,0.3])
ax10.set_ylabel("TWTT (s)",labelpad=0)


for ax in [ax20,ax21,ax22,ax23,ax24]:
    ax.set_aspect('equal')
    ax.set_xlim(-8.8,8.8)
    ax.set_ylim(-8.8,8.8)
    ax.set_xlabel("East (cm)",labelpad=0)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

ax20.set_ylabel("North (cm)",labelpad=0)
plt.setp(ax20.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax21,ax22,ax23,ax24]:
    ax.tick_params(labelleft=False)


for ax in [ax30,ax31,ax32,ax33,ax34]:
    ax.set_xlim(Tleft,Tright)
    ax.set_ylim(-70,8)
    ax.set_yticks([-60,-30,0])
    ax.tick_params(axis='y', which='major', pad=0)
ax30.set_ylabel("Up (cm)",labelpad=0)
plt.setp(ax30.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax31,ax32,ax33,ax34]:
    ax.tick_params(labelleft=False)
    ax.tick_params(labelbottom=False)
ax30.tick_params(labelbottom=False)

for ax in [ax40,ax41,ax42,ax43,ax44]:
    ax.set_ylim(1498.5,1501.5)
    ax.tick_params(axis='y', which='major', pad=0)
ax40.set_ylabel("Sound V. (m/s)",labelpad=0)
plt.setp(ax40.yaxis.get_majorticklabels(), rotation=90, va="center")
ax40.set_yticks([1499,1501])
ax40.ticklabel_format(useOffset=False)

for ax in [ax41,ax42,ax43,ax44]:
    ax.tick_params(labelleft=False)
for ax in [ax40,ax41,ax42,ax43,ax44]:
    ax.tick_params(labelbottom=False)

fmtter = dts.DateFormatter('%H:%M')
for ax in [ax30,ax40,ax50,ax31,ax41,ax51,ax32,ax42,ax52,ax33,ax43,ax53,ax34,ax44,ax54]:
    ax.xaxis.set_major_locator(dts.HourLocator(interval=1))
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')
ax10.xaxis.set_major_locator(dts.MinuteLocator(interval=30))
ax10.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

for ax in [ax10,ax50,ax51,ax52,ax53,ax54]:
    ax.xaxis.set_major_formatter(fmtter)

for ax in [ax50,ax51,ax52,ax53,ax54]:
    ax.set_ylim(-15,15)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=0)
ax50.set_ylabel("Res. (cm)",labelpad=0)
plt.setp(ax50.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax51,ax52,ax53,ax54]:
    ax.tick_params(labelleft=False)
ax52.set_xlabel("HH:MM on 2021-07-23 (UTC)",labelpad=2)

ax00.text(-168,168, "a", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax01.text(-168,168, "b", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax10.text(Tleft,0, "c", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)

ax20.text(-8.8,8.8, "d", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax21.text(-8.8,8.8, "e", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax22.text(-8.8,8.8, "f", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax23.text(-8.8,8.8, "g", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax24.text(-8.8,8.8, "h", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax30.text(Tleft,-70, "i", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax31.text(Tleft,-70, "j", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax32.text(Tleft,-70, "k", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax33.text(Tleft,-70, "l", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax34.text(Tleft,-70, "m", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)

ax40.text(Tleft,1501.5, "n", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax41.text(Tleft,1501.5, "o", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax42.text(Tleft,1501.5, "p", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax43.text(Tleft,1501.5, "q", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax44.text(Tleft,1498.5, "r", color='k', fontweight='bold',fontsize=15,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)

ax50.text(Tleft,15, "s", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax51.text(Tleft,15, "t", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax52.text(Tleft,15, "u", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax53.text(Tleft,15, "v", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax54.text(Tleft,15, "w", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

tneusss0tneusss1t=np.loadtxt(fdata)
tObs=dts.date2num(UTC_ref + tneusss0tneusss1t[:,0]*timedelta(seconds=1))
pltTWTT=ax10.plot(tObs,tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)

mkK_S1=ax41.plot([Tleft,Tright], [hmean_acous_c0,hmean_acous_c0], c='k', zorder=5,figure=fig,lw=2)
mkK_S3=ax43.plot([Tleft,Tright], [hmean_acous_c0,hmean_acous_c0], c='k', zorder=5,figure=fig,lw=2)
mkK_S4=ax44.plot([Tleft,Tright], [hmean_acous_c0_woa18,hmean_acous_c0_woa18], c='k', zorder=5,figure=fig,lw=2) 


#for non-overlapping example
for k in range(0,len(Tseg0_a)):
    selID=np.where( (tObs>=Tseg0_a[k,0]) & (tObs<Tseg0_a[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]
    mkK=ax10.plot([Tseg0_a[k,0],Tseg0_a[k,1]], [0.3,0.3], c=colors_a[k], zorder=15,figure=fig,lw=2)
    pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors_a[k], marker='o',markersize=2.5,zorder=2,figure=fig,lw=0.2)

for k in range(0,len(Tseg0)):
    mkK=ax10.plot([Tseg0[k,0],Tseg0[k,1]], [0.28,0.28], c=colors[k], zorder=15,figure=fig,lw=2)
    selID=np.where( (tObs>=Tseg0[k,0]) & (tObs<Tseg0[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]
    pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=1,zorder=15,figure=fig,lw=0.2)




Tseg0=np.append( Tseg0_a, Tseg0,axis=0)
S0_tNEUsss=np.zeros([Tseg0.shape[0],7])
S1_tNEUsss=np.zeros([Tseg0.shape[0],7])

for k in range(0,len(Tseg0)):
    selID=np.where( (tObs>=Tseg0[k,0]) & (tObs<Tseg0[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]
    ##################
    #least squares, set V/c free
    S0_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],tObs[selID] ))
    S0_est_neuc0=np.array([[refN0],[refE0],[refU0],[hmean_acous_c0]])
    S0_A0, S0_W0, S0_l0 = getAWl_freeC(S0_est_neuc0,S0_tneusss0neusss1T[:,0:13])
    S0_AtW0=sparse.csr_matrix.dot(S0_A0.T,S0_W0)
    S0_AtWA0=np.dot(S0_AtW0,S0_A0)
    S0_AtWl0=np.dot(S0_AtW0,S0_l0)
    S0_d_neuc=np.linalg.solve( S0_AtWA0, S0_AtWl0 )
    S0_est_neuc0=S0_est_neuc0+S0_d_neuc
    
    S0_slant_r0=np.sqrt( (S0_tneusss0neusss1T[:,1]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,2]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,3]-S0_est_neuc0[2,0])**2 )
    S0_slant_r1=np.sqrt( (S0_tneusss0neusss1T[:,7]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,8]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,9]-S0_est_neuc0[2,0])**2 )
    res = S0_est_neuc0[3,0]*S0_tneusss0neusss1T[:,0] - S0_slant_r0 -S0_slant_r1
    res_sigma2 = ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,1])*S0_tneusss0neusss1T[:,4]/S0_slant_r0)**2  + \
                 ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,2])*S0_tneusss0neusss1T[:,5]/S0_slant_r0)**2  + \
                 ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,3])*S0_tneusss0neusss1T[:,6]/S0_slant_r0)**2  + \
                 ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,7])*S0_tneusss0neusss1T[:,10]/S0_slant_r1)**2  + \
                 ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,8])*S0_tneusss0neusss1T[:,11]/S0_slant_r1)**2  + \
                 ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,9])*S0_tneusss0neusss1T[:,12]/S0_slant_r1)**2
    res_weights=1.0/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*twtT_thres_rms_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        S0_tneusss0neusss1T = S0_tneusss0neusss1T[sID_res,:]
        S0_A0, S0_W0, S0_l0 = getAWl_freeC(S0_est_neuc0,S0_tneusss0neusss1T[:,0:13])
        S0_AtW0=sparse.csr_matrix.dot(S0_A0.T,S0_W0)
        S0_AtWA0=np.dot(S0_AtW0,S0_A0)
        S0_AtWl0=np.dot(S0_AtW0,S0_l0)
        S0_d_neuc=np.linalg.solve( S0_AtWA0, S0_AtWl0 )
        kcount=0
        while ( ( ( np.abs(S0_d_neuc[0,0]) >1.0e-7) or (np.abs(S0_d_neuc[1,0]) >1.0e-7) or (np.abs(S0_d_neuc[2,0]) >3.0e-7) or (np.abs(S0_d_neuc[3,0]) >1.0e-10) ) and (kcount <50) ):
            S0_est_neuc0=S0_est_neuc0+S0_d_neuc
            S0_A0, S0_W0, S0_l0 = getAWl_freeC(S0_est_neuc0,S0_tneusss0neusss1T[:,0:13])
            S0_AtW0=sparse.csr_matrix.dot(S0_A0.T,S0_W0)
            S0_AtWA0=np.dot(S0_AtW0,S0_A0)
            S0_AtWl0=np.dot(S0_AtW0,S0_l0)
            S0_d_neuc=np.linalg.solve( S0_AtWA0, S0_AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            S0_slant_r0=np.sqrt( (S0_tneusss0neusss1T[:,1]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,2]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,3]-S0_est_neuc0[2,0])**2 )
            S0_slant_r1=np.sqrt( (S0_tneusss0neusss1T[:,7]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,8]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,9]-S0_est_neuc0[2,0])**2 )
            res = S0_est_neuc0[3,0]*S0_tneusss0neusss1T[:,0] - S0_slant_r0 -S0_slant_r1
            res_sigma2 = ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,1])*S0_tneusss0neusss1T[:,4]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,2])*S0_tneusss0neusss1T[:,5]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,3])*S0_tneusss0neusss1T[:,6]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,7])*S0_tneusss0neusss1T[:,10]/S0_slant_r1)**2  + \
                         ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,8])*S0_tneusss0neusss1T[:,11]/S0_slant_r1)**2  + \
                         ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,9])*S0_tneusss0neusss1T[:,12]/S0_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            S0_est_neuc0=S0_est_neuc0+S0_d_neuc
            S0_slant_r0=np.sqrt( (S0_tneusss0neusss1T[:,1]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,2]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,3]-S0_est_neuc0[2,0])**2 )
            S0_slant_r1=np.sqrt( (S0_tneusss0neusss1T[:,7]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,8]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,9]-S0_est_neuc0[2,0])**2 )
            res = S0_est_neuc0[3,0]*S0_tneusss0neusss1T[:,0] - S0_slant_r0 -S0_slant_r1
            res_sigma2 = ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,1])*S0_tneusss0neusss1T[:,4]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,2])*S0_tneusss0neusss1T[:,5]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,3])*S0_tneusss0neusss1T[:,6]/S0_slant_r0)**2  + \
                         ((S0_est_neuc0[0,0]-S0_tneusss0neusss1T[:,7])*S0_tneusss0neusss1T[:,10]/S0_slant_r1)**2  + \
                         ((S0_est_neuc0[1,0]-S0_tneusss0neusss1T[:,8])*S0_tneusss0neusss1T[:,11]/S0_slant_r1)**2  + \
                         ((S0_est_neuc0[2,0]-S0_tneusss0neusss1T[:,9])*S0_tneusss0neusss1T[:,12]/S0_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))

    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (S0_tneusss0neusss1T[:,7]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,8]-S0_est_neuc0[1,0])**2 ) /  \
                          np.sqrt( (S0_tneusss0neusss1T[:,7]-S0_est_neuc0[0,0])**2 + (S0_tneusss0neusss1T[:,8]-S0_est_neuc0[1,0])**2 + (S0_tneusss0neusss1T[:,9]-S0_est_neuc0[2,0])**2 ) )
        HZazimuth=np.zeros(S0_tneusss0neusss1T.shape[0])
        for i in range(0,S0_tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(S0_tneusss0neusss1T[i,7]-S0_est_neuc0[0,0],S0_tneusss0neusss1T[i,8]-S0_est_neuc0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        
        pltHZ=ax20.errorbar(S0_est_neuc0[1,0]*100.0,S0_est_neuc0[0,0]*100.0,xerr=100.0*e_res_wrms,yerr=100.0*n_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltU =ax30.errorbar(np.mean(S0_tneusss0neusss1T[:,13]),S0_est_neuc0[2,0]*100.0,yerr=100.0*u_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltC =ax40.plot(np.mean(S0_tneusss0neusss1T[:,13]),S0_est_neuc0[3,0],marker='o',c=colors_comb[k],markersize=2,figure=fig)
        plt_res=ax50.plot(S0_tneusss0neusss1T[:,13],0.5*res*100, c=colors_comb[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)

        S0_tNEUsss[k,0]=np.mean(S0_tneusss0neusss1T[:,13])
        S0_tNEUsss[k,1]=S0_est_neuc0[0,0]
        S0_tNEUsss[k,2]=S0_est_neuc0[1,0]
        S0_tNEUsss[k,3]=S0_est_neuc0[2,0]
        S0_tNEUsss[k,4]=n_res_wrms
        S0_tNEUsss[k,5]=e_res_wrms
        S0_tNEUsss[k,6]=u_res_wrms 



    #least squares, fix C (measured by nearby mooring)
    S1_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],tObs[selID] ))
    S1_est_neu0=np.array([[refN0],[refE0],[refU0]])
    S1_A0, S1_W0, S1_l0 = getAWl(S1_est_neu0,S1_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
    S1_AtW0=sparse.csr_matrix.dot(S1_A0.T,S1_W0)
    S1_AtWA0=np.dot(S1_AtW0,S1_A0)
    S1_AtWl0=np.dot(S1_AtW0,S1_l0)
    S1_d_neu=np.linalg.solve( S1_AtWA0, S1_AtWl0 )
    S1_est_neu0=S1_est_neu0+S1_d_neu
    
    S1_slant_r0=np.sqrt( (S1_tneusss0neusss1T[:,1]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,2]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,3]-S1_est_neu0[2,0])**2 )
    S1_slant_r1=np.sqrt( (S1_tneusss0neusss1T[:,7]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,8]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,9]-S1_est_neu0[2,0])**2 )
    res = hmean_acous_c0*S1_tneusss0neusss1T[:,0] - S1_slant_r0 -S1_slant_r1
    res_sigma2 = ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,1])*S1_tneusss0neusss1T[:,4]/S1_slant_r0)**2  + \
                 ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,2])*S1_tneusss0neusss1T[:,5]/S1_slant_r0)**2  + \
                 ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,3])*S1_tneusss0neusss1T[:,6]/S1_slant_r0)**2  + \
                 ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,7])*S1_tneusss0neusss1T[:,10]/S1_slant_r1)**2  + \
                 ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,8])*S1_tneusss0neusss1T[:,11]/S1_slant_r1)**2  + \
                 ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,9])*S1_tneusss0neusss1T[:,12]/S1_slant_r1)**2
    res_weights=1.0/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*twtT_thres_rms_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        S1_tneusss0neusss1T = S1_tneusss0neusss1T[sID_res,:]
        S1_A0, S1_W0, S1_l0 = getAWl(S1_est_neu0,S1_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
        S1_AtW0=sparse.csr_matrix.dot(S1_A0.T,S1_W0)
        S1_AtWA0=np.dot(S1_AtW0,S1_A0)
        S1_AtWl0=np.dot(S1_AtW0,S1_l0)
        S1_d_neu=np.linalg.solve( S1_AtWA0, S1_AtWl0 )
        kcount=0
        while ( ( ( np.abs(S1_d_neu[0,0]) >1.0e-7) or (np.abs(S1_d_neu[1,0]) >1.0e-7) or (np.abs(S1_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
            S1_est_neu0=S1_est_neu0+S1_d_neu
            S1_A0, S1_W0, S1_l0 = getAWl(S1_est_neu0,S1_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
            S1_AtW0=sparse.csr_matrix.dot(S1_A0.T,S1_W0)
            S1_AtWA0=np.dot(S1_AtW0,S1_A0)
            S1_AtWl0=np.dot(S1_AtW0,S1_l0)
            S1_d_neu=np.linalg.solve( S1_AtWA0, S1_AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            S1_slant_r0=np.sqrt( (S1_tneusss0neusss1T[:,1]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,2]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,3]-S1_est_neu0[2,0])**2 )
            S1_slant_r1=np.sqrt( (S1_tneusss0neusss1T[:,7]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,8]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,9]-S1_est_neu0[2,0])**2 )
            res = hmean_acous_c0*S1_tneusss0neusss1T[:,0] - S1_slant_r0 -S1_slant_r1
            res_sigma2 = ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,1])*S1_tneusss0neusss1T[:,4]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,2])*S1_tneusss0neusss1T[:,5]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,3])*S1_tneusss0neusss1T[:,6]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,7])*S1_tneusss0neusss1T[:,10]/S1_slant_r1)**2  + \
                         ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,8])*S1_tneusss0neusss1T[:,11]/S1_slant_r1)**2  + \
                         ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,9])*S1_tneusss0neusss1T[:,12]/S1_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            S1_est_neu0=S1_est_neu0+S1_d_neu
            S1_slant_r0=np.sqrt( (S1_tneusss0neusss1T[:,1]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,2]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,3]-S1_est_neu0[2,0])**2 )
            S1_slant_r1=np.sqrt( (S1_tneusss0neusss1T[:,7]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,8]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,9]-S1_est_neu0[2,0])**2 )
            res = hmean_acous_c0*S1_tneusss0neusss1T[:,0] - S1_slant_r0 -S1_slant_r1
            res_sigma2 = ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,1])*S1_tneusss0neusss1T[:,4]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,2])*S1_tneusss0neusss1T[:,5]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,3])*S1_tneusss0neusss1T[:,6]/S1_slant_r0)**2  + \
                         ((S1_est_neu0[0,0]-S1_tneusss0neusss1T[:,7])*S1_tneusss0neusss1T[:,10]/S1_slant_r1)**2  + \
                         ((S1_est_neu0[1,0]-S1_tneusss0neusss1T[:,8])*S1_tneusss0neusss1T[:,11]/S1_slant_r1)**2  + \
                         ((S1_est_neu0[2,0]-S1_tneusss0neusss1T[:,9])*S1_tneusss0neusss1T[:,12]/S1_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))

    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (S1_tneusss0neusss1T[:,7]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,8]-S1_est_neu0[1,0])**2 ) /  \
                          np.sqrt( (S1_tneusss0neusss1T[:,7]-S1_est_neu0[0,0])**2 + (S1_tneusss0neusss1T[:,8]-S1_est_neu0[1,0])**2 + (S1_tneusss0neusss1T[:,9]-S1_est_neu0[2,0])**2 ) )
        HZazimuth=np.zeros(S1_tneusss0neusss1T.shape[0])
        for i in range(0,S1_tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(S1_tneusss0neusss1T[i,7]-S1_est_neu0[0,0],S1_tneusss0neusss1T[i,8]-S1_est_neu0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        
        pltHZ=ax21.errorbar(S1_est_neu0[1,0]*100.0,S1_est_neu0[0,0]*100.0,xerr=100.0*e_res_wrms,yerr=100.0*n_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltU =ax31.errorbar(np.mean(S1_tneusss0neusss1T[:,13]),S1_est_neu0[2,0]*100.0,yerr=100.0*u_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        plt_res=ax51.plot(S1_tneusss0neusss1T[:,13],0.5*res*100, c=colors_comb[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
        S1_tNEUsss[k,0]=np.mean(S1_tneusss0neusss1T[:,13])
        S1_tNEUsss[k,1]=S1_est_neu0[0,0]
        S1_tNEUsss[k,2]=S1_est_neu0[1,0]
        S1_tNEUsss[k,3]=S1_est_neu0[2,0]
        S1_tNEUsss[k,4]=n_res_wrms
        S1_tNEUsss[k,5]=e_res_wrms
        S1_tNEUsss[k,6]=u_res_wrms 

#calculated the weighted mean and weighted standard deviation
S0_NEss=S0_tNEUsss[2:,[1,2,4,5]]
S0_weightsN=1.0/(S0_NEss[:,2]**2)
S0_sumwN=np.sum(S0_weightsN)
S0_weightsE=1.0/(S0_NEss[:,3]**2)
S0_sumwE=np.sum(S0_weightsE)
S0_wMeanN=np.sum(S0_weightsN*S0_NEss[:,0])/S0_sumwN
S0_wMeanE=np.sum(S0_weightsE*S0_NEss[:,1])/S0_sumwE
S0_wSTDN=np.sqrt( np.sum(S0_weightsN*((S0_NEss[:,0]-S0_wMeanN)**2))  / ((S0_NEss.shape[0]-1)*S0_sumwN/S0_NEss.shape[0])  )
S0_wSTDE=np.sqrt( np.sum(S0_weightsE*((S0_NEss[:,1]-S0_wMeanE)**2))  / ((S0_NEss.shape[0]-1)*S0_sumwE/S0_NEss.shape[0])  )
ax20.text(-8.5,-4.8, "std (cm)", color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax20.text(-8.5,-6.8, "North: %.1f" %(S0_wSTDN*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax20.text(-8.5,-8.8, "East: %.1f" %(S0_wSTDE*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)

S1_NEss=S1_tNEUsss[2:,[1,2,4,5]]
S1_weightsN=1.0/(S1_NEss[:,2]**2)
S1_sumwN=np.sum(S1_weightsN)
S1_weightsE=1.0/(S1_NEss[:,3]**2)
S1_sumwE=np.sum(S1_weightsE)
S1_wMeanN=np.sum(S1_weightsN*S1_NEss[:,0])/S1_sumwN
S1_wMeanE=np.sum(S1_weightsE*S1_NEss[:,1])/S1_sumwE
S1_wSTDN=np.sqrt( np.sum(S1_weightsN*((S1_NEss[:,0]-S1_wMeanN)**2))  / ((S1_NEss.shape[0]-1)*S1_sumwN/S1_NEss.shape[0])  )
S1_wSTDE=np.sqrt( np.sum(S1_weightsE*((S1_NEss[:,1]-S1_wMeanE)**2))  / ((S1_NEss.shape[0]-1)*S1_sumwE/S1_NEss.shape[0])  )
ax21.text(-8.5,-4.8, "std (cm)", color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax21.text(-8.5,-6.8, "North: %.1f" %(S1_wSTDN*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax21.text(-8.5,-8.8, "East: %.1f" %(S1_wSTDE*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)



#for segments with overlapping data
for k in range(0,len(Tseg1_a)):
    selID=np.where( (tObs>=Tseg1_a[k,0]) & (tObs<Tseg1_a[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]
    mkK=ax10.plot([Tseg1_a[k,0],Tseg1_a[k,1]], [0.01,0.01], c=colors_a[k], zorder=15,figure=fig,lw=1)
    pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors_a[k], marker='o',markersize=2.5,zorder=2,figure=fig,lw=0.2)

for k in range(0,len(Tseg1)):
    selID=np.where( (tObs>=Tseg1[k,0]) & (tObs<Tseg1[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax10.plot([Tseg1[k,0],Tseg1[k,1]], [0.1,0.1], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0.2)
    elif (k%4==1):
        mkK=ax10.plot([Tseg1[k,0],Tseg1[k,1]], [0.08,0.08], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0.2)
    elif (k%4==2):
        mkK=ax10.plot([Tseg1[k,0],Tseg1[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0.2)
    elif (k%4==3):
        mkK=ax10.plot([Tseg1[k,0],Tseg1[k,1]], [0.04,0.04], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0.2)

Tseg1=np.append( Tseg1_a, Tseg1 ,axis=0)
S2_tNEUsss=np.zeros([Tseg1.shape[0],7])
S3_tNEUsss=np.zeros([Tseg1.shape[0],7])
S4_tNEUsss=np.zeros([Tseg1.shape[0],7])

for k in range(0,len(Tseg1)):
    selID=np.where( (tObs>=Tseg1[k,0]) & (tObs<Tseg1[k,1]))[0]
    sel_tneusss0tneusss1t=tneusss0tneusss1t[selID,:]

    ##################
    #least squares, set V/c free
    S2_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],tObs[selID] ))
    S2_est_neuc0=np.array([[refN0],[refE0],[refU0],[hmean_acous_c0]])
    S2_A0, S2_W0, S2_l0 = getAWl_freeC(S2_est_neuc0,S2_tneusss0neusss1T[:,0:13])
    S2_AtW0=sparse.csr_matrix.dot(S2_A0.T,S2_W0)
    S2_AtWA0=np.dot(S2_AtW0,S2_A0)
    S2_AtWl0=np.dot(S2_AtW0,S2_l0)
    S2_d_neuc=np.linalg.solve( S2_AtWA0, S2_AtWl0 )
    S2_est_neuc0=S2_est_neuc0+S2_d_neuc
    
    S2_slant_r0=np.sqrt( (S2_tneusss0neusss1T[:,1]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,2]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,3]-S2_est_neuc0[2,0])**2 )
    S2_slant_r1=np.sqrt( (S2_tneusss0neusss1T[:,7]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,8]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,9]-S2_est_neuc0[2,0])**2 )
    res = S2_est_neuc0[3,0]*S2_tneusss0neusss1T[:,0] - S2_slant_r0 -S2_slant_r1
    res_sigma2 = ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,1])*S2_tneusss0neusss1T[:,4]/S2_slant_r0)**2  + \
                 ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,2])*S2_tneusss0neusss1T[:,5]/S2_slant_r0)**2  + \
                 ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,3])*S2_tneusss0neusss1T[:,6]/S2_slant_r0)**2  + \
                 ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,7])*S2_tneusss0neusss1T[:,10]/S2_slant_r1)**2  + \
                 ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,8])*S2_tneusss0neusss1T[:,11]/S2_slant_r1)**2  + \
                 ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,9])*S2_tneusss0neusss1T[:,12]/S2_slant_r1)**2
    res_weights=1.0/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*twtT_thres_rms_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        S2_tneusss0neusss1T = S2_tneusss0neusss1T[sID_res,:]
        S2_A0, S2_W0, S2_l0 = getAWl_freeC(S2_est_neuc0,S2_tneusss0neusss1T[:,0:13])
        S2_AtW0=sparse.csr_matrix.dot(S2_A0.T,S2_W0)
        S2_AtWA0=np.dot(S2_AtW0,S2_A0)
        S2_AtWl0=np.dot(S2_AtW0,S2_l0)
        S2_d_neuc=np.linalg.solve( S2_AtWA0, S2_AtWl0 )
        kcount=0
        while ( ( ( np.abs(S2_d_neuc[0,0]) >1.0e-7) or (np.abs(S2_d_neuc[1,0]) >1.0e-7) or (np.abs(S2_d_neuc[2,0]) >3.0e-7) or (np.abs(S2_d_neuc[3,0]) >1.0e-10) ) and (kcount <50) ):
            S2_est_neuc0=S2_est_neuc0+S2_d_neuc
            S2_A0, S2_W0, S2_l0 = getAWl_freeC(S2_est_neuc0,S2_tneusss0neusss1T[:,0:13])
            S2_AtW0=sparse.csr_matrix.dot(S2_A0.T,S2_W0)
            S2_AtWA0=np.dot(S2_AtW0,S2_A0)
            S2_AtWl0=np.dot(S2_AtW0,S2_l0)
            S2_d_neuc=np.linalg.solve( S2_AtWA0, S2_AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            S2_slant_r0=np.sqrt( (S2_tneusss0neusss1T[:,1]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,2]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,3]-S2_est_neuc0[2,0])**2 )
            S2_slant_r1=np.sqrt( (S2_tneusss0neusss1T[:,7]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,8]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,9]-S2_est_neuc0[2,0])**2 )
            res = S2_est_neuc0[3,0]*S2_tneusss0neusss1T[:,0] - S2_slant_r0 -S2_slant_r1
            res_sigma2 = ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,1])*S2_tneusss0neusss1T[:,4]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,2])*S2_tneusss0neusss1T[:,5]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,3])*S2_tneusss0neusss1T[:,6]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,7])*S2_tneusss0neusss1T[:,10]/S2_slant_r1)**2  + \
                         ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,8])*S2_tneusss0neusss1T[:,11]/S2_slant_r1)**2  + \
                         ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,9])*S2_tneusss0neusss1T[:,12]/S2_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            S2_est_neuc0=S2_est_neuc0+S2_d_neuc
            S2_slant_r0=np.sqrt( (S2_tneusss0neusss1T[:,1]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,2]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,3]-S2_est_neuc0[2,0])**2 )
            S2_slant_r1=np.sqrt( (S2_tneusss0neusss1T[:,7]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,8]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,9]-S2_est_neuc0[2,0])**2 )
            res = S2_est_neuc0[3,0]*S2_tneusss0neusss1T[:,0] - S2_slant_r0 -S2_slant_r1
            res_sigma2 = ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,1])*S2_tneusss0neusss1T[:,4]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,2])*S2_tneusss0neusss1T[:,5]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,3])*S2_tneusss0neusss1T[:,6]/S2_slant_r0)**2  + \
                         ((S2_est_neuc0[0,0]-S2_tneusss0neusss1T[:,7])*S2_tneusss0neusss1T[:,10]/S2_slant_r1)**2  + \
                         ((S2_est_neuc0[1,0]-S2_tneusss0neusss1T[:,8])*S2_tneusss0neusss1T[:,11]/S2_slant_r1)**2  + \
                         ((S2_est_neuc0[2,0]-S2_tneusss0neusss1T[:,9])*S2_tneusss0neusss1T[:,12]/S2_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))

    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (S2_tneusss0neusss1T[:,7]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,8]-S2_est_neuc0[1,0])**2 ) /  \
                          np.sqrt( (S2_tneusss0neusss1T[:,7]-S2_est_neuc0[0,0])**2 + (S2_tneusss0neusss1T[:,8]-S2_est_neuc0[1,0])**2 + (S2_tneusss0neusss1T[:,9]-S2_est_neuc0[2,0])**2 ) )
        HZazimuth=np.zeros(S2_tneusss0neusss1T.shape[0])
        for i in range(0,S2_tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(S2_tneusss0neusss1T[i,7]-S2_est_neuc0[0,0],S2_tneusss0neusss1T[i,8]-S2_est_neuc0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        
        pltHZ=ax22.errorbar(S2_est_neuc0[1,0]*100.0,S2_est_neuc0[0,0]*100.0,xerr=100.0*e_res_wrms,yerr=100.0*n_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltU =ax32.errorbar(np.mean(S2_tneusss0neusss1T[:,13]),S2_est_neuc0[2,0]*100.0,yerr=100.0*u_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltC =ax42.plot(np.mean(S2_tneusss0neusss1T[:,13]),S2_est_neuc0[3,0],marker='o',c=colors_comb[k],markersize=2,figure=fig)
        plt_res=ax52.plot(S2_tneusss0neusss1T[:,13],0.5*res*100, c=colors_comb[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)

        S2_tNEUsss[k,0]=np.mean(S2_tneusss0neusss1T[:,13])
        S2_tNEUsss[k,1]=S2_est_neuc0[0,0]
        S2_tNEUsss[k,2]=S2_est_neuc0[1,0]
        S2_tNEUsss[k,3]=S2_est_neuc0[2,0]
        S2_tNEUsss[k,4]=n_res_wrms
        S2_tNEUsss[k,5]=e_res_wrms
        S2_tNEUsss[k,6]=u_res_wrms 



    #least squares, fix C, measured by nearby mooring
    S3_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],tObs[selID] ))
    S3_est_neu0=np.array([[refN0],[refE0],[refU0]])
    S3_A0, S3_W0, S3_l0 = getAWl(S3_est_neu0,S3_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
    S3_AtW0=sparse.csr_matrix.dot(S3_A0.T,S3_W0)
    S3_AtWA0=np.dot(S3_AtW0,S3_A0)
    S3_AtWl0=np.dot(S3_AtW0,S3_l0)
    S3_d_neu=np.linalg.solve( S3_AtWA0, S3_AtWl0 )
    S3_est_neu0=S3_est_neu0+S3_d_neu
    
    S3_slant_r0=np.sqrt( (S3_tneusss0neusss1T[:,1]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,2]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,3]-S3_est_neu0[2,0])**2 )
    S3_slant_r1=np.sqrt( (S3_tneusss0neusss1T[:,7]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,8]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,9]-S3_est_neu0[2,0])**2 )
    res = hmean_acous_c0*S3_tneusss0neusss1T[:,0] - S3_slant_r0 -S3_slant_r1
    res_sigma2 = ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,1])*S3_tneusss0neusss1T[:,4]/S3_slant_r0)**2  + \
                 ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,2])*S3_tneusss0neusss1T[:,5]/S3_slant_r0)**2  + \
                 ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,3])*S3_tneusss0neusss1T[:,6]/S3_slant_r0)**2  + \
                 ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,7])*S3_tneusss0neusss1T[:,10]/S3_slant_r1)**2  + \
                 ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,8])*S3_tneusss0neusss1T[:,11]/S3_slant_r1)**2  + \
                 ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,9])*S3_tneusss0neusss1T[:,12]/S3_slant_r1)**2
    res_weights=1.0/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*twtT_thres_rms_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        S3_tneusss0neusss1T = S3_tneusss0neusss1T[sID_res,:]
        S3_A0, S3_W0, S3_l0 = getAWl(S3_est_neu0,S3_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
        S3_AtW0=sparse.csr_matrix.dot(S3_A0.T,S3_W0)
        S3_AtWA0=np.dot(S3_AtW0,S3_A0)
        S3_AtWl0=np.dot(S3_AtW0,S3_l0)
        S3_d_neu=np.linalg.solve( S3_AtWA0, S3_AtWl0 )
        kcount=0
        while ( ( ( np.abs(S3_d_neu[0,0]) >1.0e-7) or (np.abs(S3_d_neu[1,0]) >1.0e-7) or (np.abs(S3_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
            S3_est_neu0=S3_est_neu0+S3_d_neu
            S3_A0, S3_W0, S3_l0 = getAWl(S3_est_neu0,S3_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0)
            S3_AtW0=sparse.csr_matrix.dot(S3_A0.T,S3_W0)
            S3_AtWA0=np.dot(S3_AtW0,S3_A0)
            S3_AtWl0=np.dot(S3_AtW0,S3_l0)
            S3_d_neu=np.linalg.solve( S3_AtWA0, S3_AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            S3_slant_r0=np.sqrt( (S3_tneusss0neusss1T[:,1]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,2]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,3]-S3_est_neu0[2,0])**2 )
            S3_slant_r1=np.sqrt( (S3_tneusss0neusss1T[:,7]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,8]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,9]-S3_est_neu0[2,0])**2 )
            res = hmean_acous_c0*S3_tneusss0neusss1T[:,0] - S3_slant_r0 -S3_slant_r1
            res_sigma2 = ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,1])*S3_tneusss0neusss1T[:,4]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,2])*S3_tneusss0neusss1T[:,5]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,3])*S3_tneusss0neusss1T[:,6]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,7])*S3_tneusss0neusss1T[:,10]/S3_slant_r1)**2  + \
                         ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,8])*S3_tneusss0neusss1T[:,11]/S3_slant_r1)**2  + \
                         ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,9])*S3_tneusss0neusss1T[:,12]/S3_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            S3_est_neu0=S3_est_neu0+S3_d_neu
            S3_slant_r0=np.sqrt( (S3_tneusss0neusss1T[:,1]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,2]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,3]-S3_est_neu0[2,0])**2 )
            S3_slant_r1=np.sqrt( (S3_tneusss0neusss1T[:,7]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,8]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,9]-S3_est_neu0[2,0])**2 )
            res = hmean_acous_c0*S3_tneusss0neusss1T[:,0] - S3_slant_r0 -S3_slant_r1
            res_sigma2 = ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,1])*S3_tneusss0neusss1T[:,4]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,2])*S3_tneusss0neusss1T[:,5]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,3])*S3_tneusss0neusss1T[:,6]/S3_slant_r0)**2  + \
                         ((S3_est_neu0[0,0]-S3_tneusss0neusss1T[:,7])*S3_tneusss0neusss1T[:,10]/S3_slant_r1)**2  + \
                         ((S3_est_neu0[1,0]-S3_tneusss0neusss1T[:,8])*S3_tneusss0neusss1T[:,11]/S3_slant_r1)**2  + \
                         ((S3_est_neu0[2,0]-S3_tneusss0neusss1T[:,9])*S3_tneusss0neusss1T[:,12]/S3_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))

    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (S3_tneusss0neusss1T[:,7]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,8]-S3_est_neu0[1,0])**2 ) /  \
                          np.sqrt( (S3_tneusss0neusss1T[:,7]-S3_est_neu0[0,0])**2 + (S3_tneusss0neusss1T[:,8]-S3_est_neu0[1,0])**2 + (S3_tneusss0neusss1T[:,9]-S3_est_neu0[2,0])**2 ) )
        HZazimuth=np.zeros(S3_tneusss0neusss1T.shape[0])
        for i in range(0,S3_tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(S3_tneusss0neusss1T[i,7]-S3_est_neu0[0,0],S3_tneusss0neusss1T[i,8]-S3_est_neu0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        
        pltHZ=ax23.errorbar(S3_est_neu0[1,0]*100.0,S3_est_neu0[0,0]*100.0,xerr=100.0*e_res_wrms,yerr=100.0*n_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltU =ax33.errorbar(np.mean(S3_tneusss0neusss1T[:,13]),S3_est_neu0[2,0]*100.0,yerr=100.0*u_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        plt_res=ax53.plot(S3_tneusss0neusss1T[:,13],0.5*res*100, c=colors_comb[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
        S3_tNEUsss[k,0]=np.mean(S3_tneusss0neusss1T[:,13])
        S3_tNEUsss[k,1]=S3_est_neu0[0,0]
        S3_tNEUsss[k,2]=S3_est_neu0[1,0]
        S3_tNEUsss[k,3]=S3_est_neu0[2,0]
        S3_tNEUsss[k,4]=n_res_wrms
        S3_tNEUsss[k,5]=e_res_wrms
        S3_tNEUsss[k,6]=u_res_wrms 

    ###################
    ### least squares, fix C with WOA18 model in July
    S4_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],tObs[selID] ))
    S4_est_neu0=np.array([[refN0],[refE0],[refU0]])
    S4_A0, S4_W0, S4_l0 = getAWl(S4_est_neu0,S4_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0_woa18)
    S4_AtW0=sparse.csr_matrix.dot(S4_A0.T,S4_W0)
    S4_AtWA0=np.dot(S4_AtW0,S4_A0)
    S4_AtWl0=np.dot(S4_AtW0,S4_l0)
    S4_d_neu=np.linalg.solve( S4_AtWA0, S4_AtWl0 )
    S4_est_neu0=S4_est_neu0+S4_d_neu
    
    S4_slant_r0=np.sqrt( (S4_tneusss0neusss1T[:,1]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,2]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,3]-S4_est_neu0[2,0])**2 )
    S4_slant_r1=np.sqrt( (S4_tneusss0neusss1T[:,7]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,8]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,9]-S4_est_neu0[2,0])**2 )
    res = hmean_acous_c0_woa18*S4_tneusss0neusss1T[:,0] - S4_slant_r0 -S4_slant_r1
    res_sigma2 = ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,1])*S4_tneusss0neusss1T[:,4]/S4_slant_r0)**2  + \
                 ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,2])*S4_tneusss0neusss1T[:,5]/S4_slant_r0)**2  + \
                 ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,3])*S4_tneusss0neusss1T[:,6]/S4_slant_r0)**2  + \
                 ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,7])*S4_tneusss0neusss1T[:,10]/S4_slant_r1)**2  + \
                 ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,8])*S4_tneusss0neusss1T[:,11]/S4_slant_r1)**2  + \
                 ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,9])*S4_tneusss0neusss1T[:,12]/S4_slant_r1)**2
    res_weights=1.0/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*twtT_thres_rms_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        S4_tneusss0neusss1T = S4_tneusss0neusss1T[sID_res,:]
        S4_A0, S4_W0, S4_l0 = getAWl(S4_est_neu0,S4_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0_woa18)
        S4_AtW0=sparse.csr_matrix.dot(S4_A0.T,S4_W0)
        S4_AtWA0=np.dot(S4_AtW0,S4_A0)
        S4_AtWl0=np.dot(S4_AtW0,S4_l0)
        S4_d_neu=np.linalg.solve( S4_AtWA0, S4_AtWl0 )
        kcount=0
        while ( ( ( np.abs(S4_d_neu[0,0]) >1.0e-7) or (np.abs(S4_d_neu[1,0]) >1.0e-7) or (np.abs(S4_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
            S4_est_neu0=S4_est_neu0+S4_d_neu
            S4_A0, S4_W0, S4_l0 = getAWl(S4_est_neu0,S4_tneusss0neusss1T[:,0:13],sigma0,hmean_acous_c0_woa18)
            S4_AtW0=sparse.csr_matrix.dot(S4_A0.T,S4_W0)
            S4_AtWA0=np.dot(S4_AtW0,S4_A0)
            S4_AtWl0=np.dot(S4_AtW0,S4_l0)
            S4_d_neu=np.linalg.solve( S4_AtWA0, S4_AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            S4_slant_r0=np.sqrt( (S4_tneusss0neusss1T[:,1]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,2]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,3]-S4_est_neu0[2,0])**2 )
            S4_slant_r1=np.sqrt( (S4_tneusss0neusss1T[:,7]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,8]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,9]-S4_est_neu0[2,0])**2 )
            res = hmean_acous_c0_woa18*S4_tneusss0neusss1T[:,0] - S4_slant_r0 -S4_slant_r1
            res_sigma2 = ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,1])*S4_tneusss0neusss1T[:,4]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,2])*S4_tneusss0neusss1T[:,5]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,3])*S4_tneusss0neusss1T[:,6]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,7])*S4_tneusss0neusss1T[:,10]/S4_slant_r1)**2  + \
                         ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,8])*S4_tneusss0neusss1T[:,11]/S4_slant_r1)**2  + \
                         ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,9])*S4_tneusss0neusss1T[:,12]/S4_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            S4_est_neu0=S4_est_neu0+S4_d_neu
            S4_slant_r0=np.sqrt( (S4_tneusss0neusss1T[:,1]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,2]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,3]-S4_est_neu0[2,0])**2 )
            S4_slant_r1=np.sqrt( (S4_tneusss0neusss1T[:,7]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,8]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,9]-S4_est_neu0[2,0])**2 )
            res = hmean_acous_c0_woa18*S4_tneusss0neusss1T[:,0] - S4_slant_r0 -S4_slant_r1
            res_sigma2 = ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,1])*S4_tneusss0neusss1T[:,4]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,2])*S4_tneusss0neusss1T[:,5]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,3])*S4_tneusss0neusss1T[:,6]/S4_slant_r0)**2  + \
                         ((S4_est_neu0[0,0]-S4_tneusss0neusss1T[:,7])*S4_tneusss0neusss1T[:,10]/S4_slant_r1)**2  + \
                         ((S4_est_neu0[1,0]-S4_tneusss0neusss1T[:,8])*S4_tneusss0neusss1T[:,11]/S4_slant_r1)**2  + \
                         ((S4_est_neu0[2,0]-S4_tneusss0neusss1T[:,9])*S4_tneusss0neusss1T[:,12]/S4_slant_r1)**2
            res_weights=1.0/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))

    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (S4_tneusss0neusss1T[:,7]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,8]-S4_est_neu0[1,0])**2 ) /  \
                          np.sqrt( (S4_tneusss0neusss1T[:,7]-S4_est_neu0[0,0])**2 + (S4_tneusss0neusss1T[:,8]-S4_est_neu0[1,0])**2 + (S4_tneusss0neusss1T[:,9]-S4_est_neu0[2,0])**2 ) )
        HZazimuth=np.zeros(S4_tneusss0neusss1T.shape[0])
        for i in range(0,S4_tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(S4_tneusss0neusss1T[i,7]-S4_est_neu0[0,0],S4_tneusss0neusss1T[i,8]-S4_est_neu0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        
        pltHZ=ax24.errorbar(S4_est_neu0[1,0]*100.0,S4_est_neu0[0,0]*100.0,xerr=100.0*e_res_wrms,yerr=100.0*n_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        pltU =ax34.errorbar(np.mean(S4_tneusss0neusss1T[:,13]),S4_est_neu0[2,0]*100.0,yerr=100.0*u_res_wrms,fmt='o',c=colors_comb[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
        plt_res=ax54.plot(S4_tneusss0neusss1T[:,13],0.5*res*100, c=colors_comb[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
        S4_tNEUsss[k,0]=np.mean(S4_tneusss0neusss1T[:,13])
        S4_tNEUsss[k,1]=S4_est_neu0[0,0]
        S4_tNEUsss[k,2]=S4_est_neu0[1,0]
        S4_tNEUsss[k,3]=S4_est_neu0[2,0]
        S4_tNEUsss[k,4]=n_res_wrms
        S4_tNEUsss[k,5]=e_res_wrms
        S4_tNEUsss[k,6]=u_res_wrms


S2_NEss=S2_tNEUsss[2:,[1,2,4,5]]
S2_weightsN=1.0/(S2_NEss[:,2]**2)
S2_sumwN=np.sum(S2_weightsN)
S2_weightsE=1.0/(S2_NEss[:,3]**2)
S2_sumwE=np.sum(S2_weightsE)
S2_wMeanN=np.sum(S2_weightsN*S2_NEss[:,0])/S2_sumwN
S2_wMeanE=np.sum(S2_weightsE*S2_NEss[:,1])/S2_sumwE
S2_wSTDN=np.sqrt( np.sum(S2_weightsN*((S2_NEss[:,0]-S2_wMeanN)**2))  / ((S2_NEss.shape[0]-1)*S2_sumwN/S2_NEss.shape[0])  )
S2_wSTDE=np.sqrt( np.sum(S2_weightsE*((S2_NEss[:,1]-S2_wMeanE)**2))  / ((S2_NEss.shape[0]-1)*S2_sumwE/S2_NEss.shape[0])  )
ax22.text(-8.5,-4.8, "std (cm)", color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax22.text(-8.5,-6.8, "North: %.1f" %(S2_wSTDN*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax22.text(-8.5,-8.8, "East: %.1f" %(S2_wSTDE*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)

S3_NEss=S3_tNEUsss[2:,[1,2,4,5]]
S3_weightsN=1.0/(S3_NEss[:,2]**2)
S3_sumwN=np.sum(S3_weightsN)
S3_weightsE=1.0/(S3_NEss[:,3]**2)
S3_sumwE=np.sum(S3_weightsE)
S3_wMeanN=np.sum(S3_weightsN*S3_NEss[:,0])/S3_sumwN
S3_wMeanE=np.sum(S3_weightsE*S3_NEss[:,1])/S3_sumwE
S3_wSTDN=np.sqrt( np.sum(S3_weightsN*((S3_NEss[:,0]-S3_wMeanN)**2))  / ((S3_NEss.shape[0]-1)*S3_sumwN/S3_NEss.shape[0])  )
S3_wSTDE=np.sqrt( np.sum(S3_weightsE*((S3_NEss[:,1]-S3_wMeanE)**2))  / ((S3_NEss.shape[0]-1)*S3_sumwE/S3_NEss.shape[0])  )
ax23.text(-8.5,-4.8, "std (cm)", color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax23.text(-8.5,-6.8, "North: %.1f" %(S3_wSTDN*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax23.text(-8.5,-8.8, "East: %.1f" %(S3_wSTDE*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)


S4_NEss=S4_tNEUsss[2:,[1,2,4,5]]
S4_weightsN=1.0/(S4_NEss[:,2]**2)
S4_sumwN=np.sum(S4_weightsN)
S4_weightsE=1.0/(S4_NEss[:,3]**2)
S4_sumwE=np.sum(S4_weightsE)
S4_wMeanN=np.sum(S4_weightsN*S4_NEss[:,0])/S4_sumwN
S4_wMeanE=np.sum(S4_weightsE*S4_NEss[:,1])/S4_sumwE
S4_wSTDN=np.sqrt( np.sum(S4_weightsN*((S4_NEss[:,0]-S4_wMeanN)**2))  / ((S4_NEss.shape[0]-1)*S4_sumwN/S4_NEss.shape[0])  )
S4_wSTDE=np.sqrt( np.sum(S4_weightsE*((S4_NEss[:,1]-S4_wMeanE)**2))  / ((S4_NEss.shape[0]-1)*S4_sumwE/S4_NEss.shape[0])  )
ax24.text(-8.5,-4.8, "std (cm)", color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax24.text(-8.5,-6.8, "North: %.1f" %(S4_wSTDN*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)
ax24.text(-8.5,-8.8, "East: %.1f" %(S4_wSTDE*100), color='k', fontweight='normal',fontsize=8,ha='left',va='bottom', clip_on=False, zorder = 5, figure=fig)


fig.savefig('Figure8.pdf',dpi=300)

plt.show()

