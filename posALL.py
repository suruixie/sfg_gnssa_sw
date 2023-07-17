"""
Copyright (c) 2023, Surui Xie
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 




This program is used to solve for the transpoder positions for the five surveys conducted between 2021/07/23 and 2022/03/21, tested with Python 3.9.5 in macOS v11.6.6

Readers should be aware that some of the settings were used to intentionally produce exactly the same figure as Figure 9 in the published paper.

Example run:
    python posAll.py

Reference: Xie, S., Zumberge, M., Sasagawa, G., and Voytenko, D., 202x. Shallow Water Seafloor Geodesy with Wave Glider-Based GNSS-Acoustic Surveying of a Single Transponder. Submitted to Earth and Space Science

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as dts
from datetime import datetime, date, time, timedelta, timezone
from scipy import sparse


colors=['red','blue','dodgerblue','gold','lime','magenta','coral','brown','green','cyan','olive','skyblue']

refN0=0    #A priori transponder coordinate in local topocentric system (North), unit: meter
refE0=0    #A priori transponder coordinate in local topocentric system (East), unit: meter
refU0=0    #A priori transponder coordinate in local topocentric system (Up), unit: meter
sigma0=0.03                     #standard deviation used for weight calculations, unit: m
twtT_thres_rms_ratio=3          #Threshold used to remove outliers based on the residual rms

hmean_acous_c0s=[1499.3, 1502.8, 1504.7, 1504.4, 1495.4]   #Harmonic mean derived from the mooring measurement

UTC_ref = datetime(2020,1,1).replace(tzinfo=timezone.utc)     #times in seconds relative to this time

file0='data/gnssa_data_20210723.txt'
file1='data/gnssa_data_20210823_SV3_1065.txt'
file2='data/gnssa_data_20210823_SV3_1063.txt'
file3='data/gnssa_data_20211210.txt'
file4='data/gnssa_data_20220321.txt'


#segments for 2021-07-23
Tstr20210723=[ ["2021-07-23 19:34:30","2021-07-23 19:50:50"], \
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
Tseg20210723=np.zeros([len(Tstr20210723),2])
for k in range(0,len(Tstr20210723)):
    Tseg20210723[k,0]=dts.date2num(datetime.strptime(Tstr20210723[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg20210723[k,1]=dts.date2num(datetime.strptime(Tstr20210723[k][1],"%Y-%m-%d %H:%M:%S"))


#segments for 2021-08-23, SV-1065
Tstr20210823a=[ ["2021-08-23 18:00:00","2021-08-23 18:38:00"], \
        ["2021-08-23 18:06:20","2021-08-23 18:51:40"], \
        ["2021-08-23 18:27:00","2021-08-23 18:55:00"], \
        ["2021-08-23 18:31:30","2021-08-23 19:12:20"], \
        ["2021-08-23 18:48:20","2021-08-23 19:18:00"]]
Tseg20210823a=np.zeros([len(Tstr20210823a),2])
for k in range(0,len(Tstr20210823a)):
    Tseg20210823a[k,0]=dts.date2num(datetime.strptime(Tstr20210823a[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg20210823a[k,1]=dts.date2num(datetime.strptime(Tstr20210823a[k][1],"%Y-%m-%d %H:%M:%S"))
  
Tstr20210823b=[ ["2021-08-23 19:45:00","2021-08-23 20:03:50"], \
                ["2021-08-23 19:50:30","2021-08-23 20:08:20"], \
                ["2021-08-23 19:54:30","2021-08-23 20:12:30"], \
                ["2021-08-23 19:59:10","2021-08-23 20:17:30"], \
                ["2021-08-23 20:04:10","2021-08-23 20:22:00"], \
                ["2021-08-23 20:08:20","2021-08-23 20:26:10"], \
                ["2021-08-23 20:12:30","2021-08-23 20:30:20"], \
                ["2021-08-23 20:17:20","2021-08-23 20:35:20"], \
                ["2021-08-23 20:22:00","2021-08-23 20:39:50"], \
                ["2021-08-23 20:26:00","2021-08-23 20:43:40"] ]
Tseg20210823b=np.zeros([len(Tstr20210823b),2])
for k in range(0,len(Tstr20210823b)):
    Tseg20210823b[k,0]=dts.date2num(datetime.strptime(Tstr20210823b[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg20210823b[k,1]=dts.date2num(datetime.strptime(Tstr20210823b[k][1],"%Y-%m-%d %H:%M:%S"))

#segments for 2021-12-10
Tstr20211210=[ ["2021-12-10 18:32:00","2021-12-10 19:20:00"], \
               ["2021-12-10 18:50:00","2021-12-10 19:40:00"], \
               ["2021-12-10 19:10:00","2021-12-10 20:00:00"], \
               ["2021-12-10 19:30:00","2021-12-10 20:20:00"], \
               ["2021-12-10 19:50:00","2021-12-10 20:40:00"] ]
Tseg20211210=np.zeros([len(Tstr20211210),2])
for k in range(0,len(Tstr20211210)):
    Tseg20211210[k,0]=dts.date2num(datetime.strptime(Tstr20211210[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg20211210[k,1]=dts.date2num(datetime.strptime(Tstr20211210[k][1],"%Y-%m-%d %H:%M:%S"))

#segments for 2022-03-21

Tstr20220321=[ ["2022-03-21 17:40:00","2022-03-21 18:05:00"], \
               ["2022-03-21 17:55:00","2022-03-21 18:20:00"], \
               ["2022-03-21 18:10:00","2022-03-21 18:35:00"], \
               ["2022-03-21 18:25:00","2022-03-21 18:50:00"], \
               ["2022-03-21 18:40:00","2022-03-21 19:30:00"] ]
Tseg20220321=np.zeros([len(Tstr20220321),2])
for k in range(0,len(Tstr20220321)):
    Tseg20220321[k,0]=dts.date2num(datetime.strptime(Tstr20220321[k][0],"%Y-%m-%d %H:%M:%S"))
    Tseg20220321[k,1]=dts.date2num(datetime.strptime(Tstr20220321[k][1],"%Y-%m-%d %H:%M:%S"))



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

def getAWl_freeC(neuc0,tneusss0neusss1,apri_sigma):
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
    Weights=sparse.spdiags(apri_sigma**2/Obs_sigma2,0,tneusss0neusss1.shape[0],tneusss0neusss1.shape[0])
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
    
    
def lsq_freeC(neuc0,tneusss0neusss1T,apri_sigma,del_ratio):
    est_neuc0=np.copy(neuc0)
    A0, W0, l0 = getAWl_freeC(est_neuc0,tneusss0neusss1T[:,0:13],apri_sigma)
    AtW0=sparse.csr_matrix.dot(A0.T,W0)
    AtWA0=np.dot(AtW0,A0)
    AtWl0=np.dot(AtW0,l0)
    d_neuc=np.linalg.solve( AtWA0, AtWl0 )
    est_neuc0=est_neuc0+d_neuc
    r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neuc0[2,0])**2 )
    r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neuc0[2,0])**2 )
    res = est_neuc0[3,0]*tneusss0neusss1T[:,0] - r0 -r1
    res_sigma2 = ((est_neuc0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                 ((est_neuc0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                 ((est_neuc0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                 ((est_neuc0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                 ((est_neuc0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                 ((est_neuc0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
    res_weights=apri_sigma**2/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*del_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        tneusss0neusss1T = tneusss0neusss1T[sID_res,:]
        A0, W0, l0 = getAWl_freeC(est_neuc0,tneusss0neusss1T[:,0:13],apri_sigma)
        AtW0=sparse.csr_matrix.dot(A0.T,W0)
        AtWA0=np.dot(AtW0,A0)
        AtWl0=np.dot(AtW0,l0)
        d_neuc=np.linalg.solve( AtWA0, AtWl0 )
        kcount=0
        while ( ( ( np.abs(d_neuc[0,0]) >1.0e-8) or (np.abs(d_neuc[1,0]) >1.0e-8) or (np.abs(d_neuc[2,0]) >3.0e-8) or (np.abs(d_neuc[3,0]) >1.0e-11) ) and (kcount <50) ):
            est_neuc0=est_neuc0+d_neuc
            A0, W0, l0 = getAWl_freeC(est_neuc0,tneusss0neusss1T[:,0:13],apri_sigma)
            AtW0=sparse.csr_matrix.dot(A0.T,W0)
            AtWA0=np.dot(AtW0,A0)
            AtWl0=np.dot(AtW0,l0)
            d_neuc=np.linalg.solve( AtWA0, AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neuc0[2,0])**2 )
            r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neuc0[2,0])**2 )
            res = est_neuc0[3,0]*tneusss0neusss1T[:,0] - r0 -r1
            res_sigma2 = ((est_neuc0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                         ((est_neuc0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                         ((est_neuc0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                         ((est_neuc0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                         ((est_neuc0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                         ((est_neuc0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
            res_weights=apri_sigma**2/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            est_neuc0=est_neuc0+d_neuc
            r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neuc0[2,0])**2 )
            r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neuc0[2,0])**2 )
            res = est_neuc0[3,0]*tneusss0neusss1T[:,0] - r0 -r1
            res_sigma2 = ((est_neuc0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                         ((est_neuc0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                         ((est_neuc0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                         ((est_neuc0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                         ((est_neuc0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                         ((est_neuc0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
            res_weights=apri_sigma**2/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (tneusss0neusss1T[:,7]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neuc0[1,0])**2 ) /  \
                          np.sqrt( (tneusss0neusss1T[:,7]-est_neuc0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neuc0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neuc0[2,0])**2 ) )
        HZazimuth=np.zeros(tneusss0neusss1T.shape[0])
        for i in range(0,tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(tneusss0neusss1T[i,7]-est_neuc0[0,0],tneusss0neusss1T[i,8]-est_neuc0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        return np.mean(tneusss0neusss1T[:,13]), est_neuc0[0,0], est_neuc0[1,0],est_neuc0[2,0], n_res_wrms,e_res_wrms,u_res_wrms, est_neuc0[3,0], np.column_stack((tneusss0neusss1T[:,13],0.5*res))        
    else:
        print ("No sufficient data for robust position estimation")


def lsq_fixC(neu0,tneusss0neusss1T,apri_sigma,soundv,del_ratio):
    est_neu0=np.copy(neu0)
    A0, W0, l0 = getAWl(est_neu0,tneusss0neusss1T[:,0:13],apri_sigma,soundv)
    AtW0=sparse.csr_matrix.dot(A0.T,W0)
    AtWA0=np.dot(AtW0,A0)
    AtWl0=np.dot(AtW0,l0)
    d_neu=np.linalg.solve( AtWA0, AtWl0 )
    est_neu0=est_neu0+d_neu
    r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neu0[2,0])**2 )
    r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neu0[2,0])**2 )
    res = soundv*tneusss0neusss1T[:,0] - r0 -r1
    res_sigma2 = ((est_neu0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                 ((est_neu0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                 ((est_neu0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                 ((est_neu0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                 ((est_neu0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                 ((est_neu0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
    res_weights=apri_sigma**2/res_sigma2
    res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    while ( (np.max(abs(res))>(res_wrms*del_ratio)) & (len(res)>=3) ):
        sID_res=np.where( abs(res)<np.max(abs(res)) )[0]
        tneusss0neusss1T = tneusss0neusss1T[sID_res,:]
        A0, W0, l0 = getAWl(est_neu0,tneusss0neusss1T[:,0:13],apri_sigma,soundv)
        AtW0=sparse.csr_matrix.dot(A0.T,W0)
        AtWA0=np.dot(AtW0,A0)
        AtWl0=np.dot(AtW0,l0)
        d_neu=np.linalg.solve( AtWA0, AtWl0 )
        kcount=0
        while ( ( ( np.abs(d_neu[0,0]) >1.0e-8) or (np.abs(d_neu[1,0]) >1.0e-8) or (np.abs(d_neu[2,0]) >3.0e-8) ) and (kcount <50) ):
            est_neu0=est_neu0+d_neu
            A0, W0, l0 = getAWl(est_neu0,tneusss0neusss1T[:,0:13],apri_sigma,soundv)
            AtW0=sparse.csr_matrix.dot(A0.T,W0)
            AtWA0=np.dot(AtW0,A0)
            AtWl0=np.dot(AtW0,l0)
            d_neu=np.linalg.solve( AtWA0, AtWl0 )
            kcount=kcount+1 
        if (kcount==50):
            r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neu0[2,0])**2 )
            r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neu0[2,0])**2 )
            res = soundv*tneusss0neusss1T[:,0] - r0 -r1
            res_sigma2 = ((est_neu0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                         ((est_neu0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                         ((est_neu0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                         ((est_neu0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                         ((est_neu0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                         ((est_neu0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
            res_weights=apri_sigma**2/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
            print ('No convergence after 50 iteration, delete measurements with the largest residual and try again')
        else:
            est_neu0=est_neu0+d_neu
            r0=np.sqrt( (tneusss0neusss1T[:,1]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,2]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,3]-est_neu0[2,0])**2 )
            r1=np.sqrt( (tneusss0neusss1T[:,7]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neu0[2,0])**2 )
            res = soundv*tneusss0neusss1T[:,0] - r0 -r1
            res_sigma2 = ((est_neu0[0,0]-tneusss0neusss1T[:,1])*tneusss0neusss1T[:,4]/r0)**2  + \
                         ((est_neu0[1,0]-tneusss0neusss1T[:,2])*tneusss0neusss1T[:,5]/r0)**2  + \
                         ((est_neu0[2,0]-tneusss0neusss1T[:,3])*tneusss0neusss1T[:,6]/r0)**2  + \
                         ((est_neu0[0,0]-tneusss0neusss1T[:,7])*tneusss0neusss1T[:,10]/r1)**2  + \
                         ((est_neu0[1,0]-tneusss0neusss1T[:,8])*tneusss0neusss1T[:,11]/r1)**2  + \
                         ((est_neu0[2,0]-tneusss0neusss1T[:,9])*tneusss0neusss1T[:,12]/r1)**2
            res_weights=apri_sigma**2/res_sigma2
            res_wrms=np.sqrt(np.sum(res_weights*(res**2))/np.sum(res_weights))
    if (len(res)>=3):
        Utheta=np.arccos( np.sqrt( (tneusss0neusss1T[:,7]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neu0[1,0])**2 ) /  \
                          np.sqrt( (tneusss0neusss1T[:,7]-est_neu0[0,0])**2 + (tneusss0neusss1T[:,8]-est_neu0[1,0])**2 + (tneusss0neusss1T[:,9]-est_neu0[2,0])**2 ) )
        HZazimuth=np.zeros(tneusss0neusss1T.shape[0])
        for i in range(0,tneusss0neusss1T.shape[0]):
            HZazimuth[i]=vecNvecE2AZ(tneusss0neusss1T[i,7]-est_neu0[0,0],tneusss0neusss1T[i,8]-est_neu0[1,0])*np.pi/180.0
        u_res=res*0.5*np.cos(Utheta)
        n_res=res*0.5*np.cos(HZazimuth)
        e_res=res*0.5*np.sin(HZazimuth)
        u_res_wrms=np.sqrt(np.sum(res_weights*(u_res**2))/np.sum(res_weights))
        n_res_wrms=np.sqrt(np.sum(res_weights*(n_res**2))/np.sum(res_weights))
        e_res_wrms=np.sqrt(np.sum(res_weights*(e_res**2))/np.sum(res_weights))
        return np.mean(tneusss0neusss1T[:,13]), est_neu0[0,0], est_neu0[1,0],est_neu0[2,0], n_res_wrms,e_res_wrms,u_res_wrms, np.column_stack((tneusss0neusss1T[:,13],0.5*res))        
    else:
        print ("No sufficient data for robust position estimation")



fig = plt.figure(figsize=(6.5,7))
gs = gridspec.GridSpec(8, 5, height_ratios=[2,0.2,1.2,0.2,2,0.2,1,1], width_ratios=[1,1,1,1,1])
gs.update(hspace=0.1,wspace=0.07)
fig.subplots_adjust(left=0.06, bottom=0.05, right=0.975, top=0.98)

ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1], sharey=ax00)
ax02 = fig.add_subplot(gs[0,2], sharey=ax00)
ax03 = fig.add_subplot(gs[0,3], sharey=ax00)
ax04 = fig.add_subplot(gs[0,4], sharey=ax00)

ax10 = fig.add_subplot(gs[2,0])
ax11 = fig.add_subplot(gs[2,1], sharey=ax10)
ax12 = fig.add_subplot(gs[2,2], sharey=ax10)
ax13 = fig.add_subplot(gs[2,3], sharey=ax10)
ax14 = fig.add_subplot(gs[2,4], sharey=ax10)

ax20 = fig.add_subplot(gs[4,0])
ax21 = fig.add_subplot(gs[4,1], sharey=ax20)
ax22 = fig.add_subplot(gs[4,2], sharey=ax20)
ax23 = fig.add_subplot(gs[4,3], sharey=ax20)
ax24 = fig.add_subplot(gs[4,4], sharey=ax20)


ax30 = fig.add_subplot(gs[6,0], sharex=ax10)
ax31 = fig.add_subplot(gs[6,1], sharex=ax11, sharey=ax30)
ax32 = fig.add_subplot(gs[6,2], sharex=ax12, sharey=ax30)
ax33 = fig.add_subplot(gs[6,3], sharex=ax13, sharey=ax30)
ax34 = fig.add_subplot(gs[6,4], sharex=ax14, sharey=ax30)

ax40 = fig.add_subplot(gs[7,0], sharex=ax10)
ax41 = fig.add_subplot(gs[7,1], sharex=ax11, sharey=ax40)
ax42 = fig.add_subplot(gs[7,2], sharex=ax12, sharey=ax40)
ax43 = fig.add_subplot(gs[7,3], sharex=ax13, sharey=ax40)
ax44 = fig.add_subplot(gs[7,4], sharex=ax14, sharey=ax40)

ax00.set_title("2021-07-23\nSV-1065", wrap=True, fontweight="bold", horizontalalignment="center", fontsize=9)
ax01.set_title("2021-08-23\nSV-1065", wrap=True, fontweight="bold", horizontalalignment="center", fontsize=9)
ax02.set_title("2021-08-23\nSV-1063", wrap=True, fontweight="bold", horizontalalignment="center", fontsize=9)
ax03.set_title("2021-12-10\nSV-1049", wrap=True, fontweight="bold", horizontalalignment="center", fontsize=9)
ax04.set_title("2022-03-21\nSV-1065", wrap=True, fontweight="bold", horizontalalignment="center", fontsize=9)


for ax in [ax00,ax01,ax02,ax03,ax04]:
    ax.set_aspect('equal')
    ax.set_xlim(-170,170)
    ax.set_ylim(-170,170)
    ax.set_xticks([-100,0,100])
    ax.set_yticks([-100,0,100])
    ax.set_xlabel("East (m)",labelpad=0)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')
ax00.set_ylabel("North (cm)",labelpad=0)
plt.setp(ax00.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax01,ax02,ax03,ax04]:
    ax.tick_params(labelleft=False)

for ax in [ax10,ax11,ax12,ax13,ax14]:
    ax.set_ylim(0,0.43)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

plt.setp(ax10.yaxis.get_majorticklabels(), rotation=90, va="center")
ax10.set_ylabel("TWTT (s)",labelpad=0)
for ax in [ax11,ax12,ax13,ax14]:
    ax.tick_params(labelleft=False)

for ax in [ax20,ax21,ax22,ax23,ax24]:
    ax.set_aspect('equal')
    ax.set_xlim(-15.6,15.6)
    ax.set_ylim(-15.6,15.6)
    ax.set_xlabel("East (cm)",labelpad=0)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=0)
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

ax20.set_ylabel("North (cm)",labelpad=0)
plt.setp(ax20.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax21,ax22,ax23,ax24]:
    ax.tick_params(labelleft=False)


for ax in [ax30,ax31,ax32,ax33,ax34]:
    ax.set_ylim(-85,10)
    ax.set_yticks([-80,-40,0])
    ax.tick_params(axis='y', which='major', pad=0)
ax30.set_ylabel("Up (cm)",labelpad=0)
plt.setp(ax30.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax31,ax32,ax33,ax34]:
    ax.tick_params(labelleft=False)
    ax.tick_params(labelbottom=False)
ax30.tick_params(labelbottom=False)


fmtter = dts.DateFormatter('%H:%M')
for ax in [ax30,ax40,ax31,ax41,ax32,ax42,ax33,ax43,ax34,ax44]:
    ax.xaxis.set_major_locator(dts.HourLocator(interval=1))
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')
ax10.xaxis.set_major_locator(dts.MinuteLocator(interval=30))
ax10.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')

for ax in [ax10,ax11,ax12,ax13,ax14,ax40,ax41,ax42,ax43,ax44]:
    ax.xaxis.set_major_formatter(fmtter)
    
    
for ax in [ax10,ax11,ax12,ax13,ax14,ax30,ax31,ax32,ax33,ax34,ax40,ax41,ax42,ax43,ax44]:
    ax.xaxis.set_major_locator(dts.HourLocator(interval=1))
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls=':')
for ax in [ax10,ax11,ax12,ax13,ax14,ax40,ax41,ax42,ax43,ax44]:
    ax.xaxis.set_major_formatter(fmtter)

ticks=[ dts.date2num(datetime.strptime(str(202108232000),"%Y%m%d%H%M")),
                 dts.date2num(datetime.strptime(str(202108232030),"%Y%m%d%H%M"))]            
for ax in [ax12,ax42]:
    ax.set_xticks(ticks)

for ax in [ax40,ax41,ax42,ax43,ax44]:
    ax.set_ylim(-33,33)
    ax.set_yticks([-20,0,20])
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(axis='y', which='major', pad=0)
ax40.set_ylabel("Res. (cm)",labelpad=0)
plt.setp(ax40.yaxis.get_majorticklabels(), rotation=90, va="center")
for ax in [ax41,ax42,ax43,ax44]:
    ax.tick_params(labelleft=False)

ax10.set_xlabel("2021-07-23",labelpad=0)
ax11.set_xlabel("2021-08-23",labelpad=0)
ax12.set_xlabel("2021-08-23",labelpad=0)
ax13.set_xlabel("2021-12-10",labelpad=0)
ax14.set_xlabel("2022-03-21",labelpad=0)

ax40.set_xlabel("2021-07-23",labelpad=0)
ax41.set_xlabel("2021-08-23",labelpad=0)
ax42.set_xlabel("2021-08-23",labelpad=0)
ax43.set_xlabel("2021-12-10",labelpad=0)
ax44.set_xlabel("2022-03-21",labelpad=0)

ax00.text(-170,170, "a", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax01.text(-170,170, "b", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax02.text(-170,170, "c", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax03.text(-170,170, "d", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax04.text(-170,170, "e", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

ax20.text(-15.6,15.6, "k", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax21.text(-15.6,15.6, "l", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax22.text(-15.6,15.6, "m", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax23.text(-15.6,15.6, "n", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax24.text(-15.6,15.6, "o", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)


##########> Plot TWTT
f0_tneusss0tneusss1t=np.loadtxt(file0)
f0_tObs=dts.date2num(UTC_ref + f0_tneusss0tneusss1t[:,0]*timedelta(seconds=1))
f0_pltTWTT=ax10.plot(f0_tObs,f0_tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)

f1_tneusss0tneusss1t=np.loadtxt(file1)
f1_tObs=dts.date2num(UTC_ref + f1_tneusss0tneusss1t[:,0]*timedelta(seconds=1))
f1_pltTWTT=ax11.plot(f1_tObs,f1_tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)

f2_tneusss0tneusss1t=np.loadtxt(file2)
f2_tObs=dts.date2num(UTC_ref + f2_tneusss0tneusss1t[:,0]*timedelta(seconds=1))
f2_pltTWTT=ax12.plot(f2_tObs,f2_tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)

f3_tneusss0tneusss1t=np.loadtxt(file3)
f3_tObs=dts.date2num(UTC_ref + f3_tneusss0tneusss1t[:,0]*timedelta(seconds=1))
f3_pltTWTT=ax13.plot(f3_tObs,f3_tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)

f4_tneusss0tneusss1t=np.loadtxt(file4)
f4_tObs=dts.date2num(UTC_ref + f4_tneusss0tneusss1t[:,0]*timedelta(seconds=1))
f4_pltTWTT=ax14.plot(f4_tObs,f4_tneusss0tneusss1t[:,14], c='k', marker='o',markersize=0.6,figure=fig,lw=0,zorder=10)
##########< end Plotting TWTT

ax10.text(f0_tObs[0]-0.003,0.43, "f", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax11.text(f1_tObs[0]-0.003,0.43, "g", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax12.text(f2_tObs[0]-0.001,0.43, "h", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax13.text(f3_tObs[0]-0.003,0.43, "i", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax14.text(f4_tObs[0]-0.003,0.43, "j", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

ax30.text(f0_tObs[0]-0.003,10, "p", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax31.text(f1_tObs[0]-0.003,10, "q", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax32.text(f2_tObs[0]-0.001,10, "r", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax33.text(f3_tObs[0]-0.003,10, "s", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax34.text(f4_tObs[0]-0.003,10, "t", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

ax40.text(f0_tObs[0]-0.003,33, "u", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax41.text(f1_tObs[0]-0.003,33, "v", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax42.text(f2_tObs[0]-0.001,33, "w", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax43.text(f3_tObs[0]-0.003,33, "x", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax44.text(f4_tObs[0]-0.003,33, "y", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)


#For test on 2021-07-23
f0_tNEUsss=np.zeros([Tseg20210723.shape[0],7])
f0_tNEUsssC_freeC=np.zeros([Tseg20210723.shape[0],8])

f0_counts=np.zeros(Tseg20210723.shape[0])
for k in range(0,len(Tseg20210723)):
    selID=np.where( (f0_tObs>=Tseg20210723[k,0]) & (f0_tObs<Tseg20210723[k,1]))[0]
    sel_tneusss0tneusss1t=f0_tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax10.plot([Tseg20210723[k,0],Tseg20210723[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0)
    elif (k%4==1):
        mkK=ax10.plot([Tseg20210723[k,0],Tseg20210723[k,1]], [0.045,0.045], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0)
    elif (k%4==2):
        mkK=ax10.plot([Tseg20210723[k,0],Tseg20210723[k,1]], [0.03,0.03], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0)
    elif (k%4==3):
        mkK=ax10.plot([Tseg20210723[k,0],Tseg20210723[k,1]], [0.015,0.015], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax00.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0)

    sel_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],f0_tObs[selID] ))
    pos_neu0=np.array([[refN0],[refE0],[refU0]])
    f0_tNEUsss[k,0],f0_tNEUsss[k,1],f0_tNEUsss[k,2],f0_tNEUsss[k,3],f0_tNEUsss[k,4],f0_tNEUsss[k,5],f0_tNEUsss[k,6], TRes = lsq_fixC(pos_neu0,sel_tneusss0neusss1T,sigma0,hmean_acous_c0s[0],twtT_thres_rms_ratio)        
    pltHZ=ax20.errorbar(f0_tNEUsss[k,2]*100.0,f0_tNEUsss[k,1]*100.0,xerr=100.0*f0_tNEUsss[k,5],yerr=100.0*f0_tNEUsss[k,4],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    pltU =ax30.errorbar(f0_tNEUsss[k,0],f0_tNEUsss[k,3]*100.0,yerr=100.0*f0_tNEUsss[k,6],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    plt_res=ax40.plot(TRes[:,0],TRes[:,1]*100, c=colors[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
    f0_counts[k]=TRes.shape[0]

    pos_neuc0_freeC=np.array([[refN0],[refE0],[refU0],[1500.0]])
    f0_tNEUsssC_freeC[k,0],f0_tNEUsssC_freeC[k,1],f0_tNEUsssC_freeC[k,2],f0_tNEUsssC_freeC[k,3],f0_tNEUsssC_freeC[k,4],f0_tNEUsssC_freeC[k,5],f0_tNEUsssC_freeC[k,6],f0_tNEUsssC_freeC[k,7], TRes_freeC = lsq_freeC(pos_neuc0_freeC,sel_tneusss0neusss1T,sigma0,twtT_thres_rms_ratio)        
    plt_res_freeC=ax40.plot(TRes_freeC[:,0],TRes_freeC[:,1]*100, c='k', marker='o',markersize=0.5,zorder=50,figure=fig,lw=0)

f0_pltNE_freeC=ax20.plot(f0_tNEUsssC_freeC[:,2]*100.0,f0_tNEUsssC_freeC[:,1]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)
f0_pltU_freeC=ax30.plot(f0_tNEUsssC_freeC[:,0],f0_tNEUsssC_freeC[:,3]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)

#For test on 2021-08-23 SV-1065
f1_tNEUsssC_freeC=np.zeros([Tseg20210823a.shape[0],8])
f1_tNEUsss=np.zeros([Tseg20210823a.shape[0],7])
f1_counts=np.zeros(Tseg20210823a.shape[0])

for k in range(0,len(Tseg20210823a)):
    selID=np.where( (f1_tObs>=Tseg20210823a[k,0]) & (f1_tObs<Tseg20210823a[k,1]) & (np.sqrt(f1_tneusss0tneusss1t[:,1]**2 + f1_tneusss0tneusss1t[:,2]**2)<=100.0) )[0]
    sel_tneusss0tneusss1t=f1_tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax11.plot([Tseg20210823a[k,0],Tseg20210823a[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0)
    elif (k%4==1):
        mkK=ax11.plot([Tseg20210823a[k,0],Tseg20210823a[k,1]], [0.045,0.045], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0)
    elif (k%4==2):
        mkK=ax11.plot([Tseg20210823a[k,0],Tseg20210823a[k,1]], [0.03,0.03], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0)
    elif (k%4==3):
        mkK=ax11.plot([Tseg20210823a[k,0],Tseg20210823a[k,1]], [0.015,0.015], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax01.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0)

    sel_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],f1_tObs[selID] ))
    pos_neu0=np.array([[refN0],[refE0],[refU0]])
    f1_tNEUsss[k,0],f1_tNEUsss[k,1],f1_tNEUsss[k,2],f1_tNEUsss[k,3],f1_tNEUsss[k,4],f1_tNEUsss[k,5],f1_tNEUsss[k,6], TRes = lsq_fixC(pos_neu0,sel_tneusss0neusss1T,sigma0,hmean_acous_c0s[1],twtT_thres_rms_ratio)        
    pltHZ=ax21.errorbar(f1_tNEUsss[k,2]*100.0,f1_tNEUsss[k,1]*100.0,xerr=100.0*f1_tNEUsss[k,5],yerr=100.0*f1_tNEUsss[k,4],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    pltU =ax31.errorbar(f1_tNEUsss[k,0],f1_tNEUsss[k,3]*100.0,yerr=100.0*f1_tNEUsss[k,6],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    plt_res=ax41.plot(TRes[:,0],TRes[:,1]*100, c=colors[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
    f1_counts[k]=TRes.shape[0]


    pos_neuc0_freeC=np.array([[refN0],[refE0],[refU0],[1500.0]])
    f1_tNEUsssC_freeC[k,0],f1_tNEUsssC_freeC[k,1],f1_tNEUsssC_freeC[k,2],f1_tNEUsssC_freeC[k,3],f1_tNEUsssC_freeC[k,4],f1_tNEUsssC_freeC[k,5],f1_tNEUsssC_freeC[k,6],f1_tNEUsssC_freeC[k,7], TRes_freeC = lsq_freeC(pos_neuc0_freeC,sel_tneusss0neusss1T,sigma0,twtT_thres_rms_ratio)        
    plt_res_freeC=ax41.plot(TRes_freeC[:,0],TRes_freeC[:,1]*100, c='k', marker='o',markersize=0.5,zorder=50,figure=fig,lw=0)

f1_pltNE_freeC=ax21.plot(f1_tNEUsssC_freeC[:,2]*100.0,f1_tNEUsssC_freeC[:,1]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)
f1_pltU_freeC=ax31.plot(f1_tNEUsssC_freeC[:,0],f1_tNEUsssC_freeC[:,3]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)


#For test on 2021-08-23 SV-1063
f2_tNEUsssC_freeC=np.zeros([Tseg20210823b.shape[0],8])

f2_tNEUsss=np.zeros([Tseg20210823b.shape[0],7])
f2_counts=np.zeros(Tseg20210823b.shape[0])

for k in range(0,len(Tseg20210823b)):
    selID=np.where( (f2_tObs>=Tseg20210823b[k,0]) & (f2_tObs<Tseg20210823b[k,1]) & (np.sqrt(f2_tneusss0tneusss1t[:,1]**2 + f2_tneusss0tneusss1t[:,2]**2)<=170.0) )[0]
    sel_tneusss0tneusss1t=f2_tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax12.plot([Tseg20210823b[k,0],Tseg20210823b[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax02.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0)
    elif (k%4==1):
        mkK=ax12.plot([Tseg20210823b[k,0],Tseg20210823b[k,1]], [0.045,0.045], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax02.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0)
    elif (k%4==2):
        mkK=ax12.plot([Tseg20210823b[k,0],Tseg20210823b[k,1]], [0.03,0.03], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax02.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0)
    elif (k%4==3):
        mkK=ax12.plot([Tseg20210823b[k,0],Tseg20210823b[k,1]], [0.015,0.015], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax02.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0)

    sel_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],f2_tObs[selID] ))
    pos_neu0=np.array([[refN0],[refE0],[refU0]])

    f2_tNEUsss[k,0],f2_tNEUsss[k,1],f2_tNEUsss[k,2],f2_tNEUsss[k,3],f2_tNEUsss[k,4],f2_tNEUsss[k,5],f2_tNEUsss[k,6], TRes = lsq_fixC(pos_neu0,sel_tneusss0neusss1T,sigma0,hmean_acous_c0s[2],twtT_thres_rms_ratio)        
    pltHZ=ax22.errorbar(f2_tNEUsss[k,2]*100.0,f2_tNEUsss[k,1]*100.0,xerr=100.0*f2_tNEUsss[k,5],yerr=100.0*f2_tNEUsss[k,4],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    pltU =ax32.errorbar(f2_tNEUsss[k,0],f2_tNEUsss[k,3]*100.0,yerr=100.0*f2_tNEUsss[k,6],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    plt_res=ax42.plot(TRes[:,0],TRes[:,1]*100, c=colors[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
    f2_counts[k]=TRes.shape[0]

    pos_neuc0_freeC=np.array([[refN0],[refE0],[refU0],[1500.0]])
    f2_tNEUsssC_freeC[k,0],f2_tNEUsssC_freeC[k,1],f2_tNEUsssC_freeC[k,2],f2_tNEUsssC_freeC[k,3],f2_tNEUsssC_freeC[k,4],f2_tNEUsssC_freeC[k,5],f2_tNEUsssC_freeC[k,6],f2_tNEUsssC_freeC[k,7], TRes_freeC = lsq_freeC(pos_neuc0_freeC,sel_tneusss0neusss1T,sigma0,twtT_thres_rms_ratio)        
    plt_res_freeC=ax42.plot(TRes_freeC[:,0],TRes_freeC[:,1]*100, c='k', marker='o',markersize=0.5,zorder=50,figure=fig,lw=0)

f2_pltNE_freeC=ax22.plot(f2_tNEUsssC_freeC[:,2]*100.0,f2_tNEUsssC_freeC[:,1]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)
f2_pltU_freeC=ax32.plot(f2_tNEUsssC_freeC[:,0],f2_tNEUsssC_freeC[:,3]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)


#For test on 2021-12-10
f3_tNEUsssC_freeC=np.zeros([Tseg20211210.shape[0],8])
f3_tNEUsss=np.zeros([Tseg20211210.shape[0],7])

f3_counts=np.zeros(Tseg20211210.shape[0])

for k in range(0,len(Tseg20211210)):
    selID=np.where( (f3_tObs>=Tseg20211210[k,0]) & (f3_tObs<Tseg20211210[k,1]) & (np.sqrt(f3_tneusss0tneusss1t[:,1]**2 + f3_tneusss0tneusss1t[:,2]**2)<=120.0) )[0]

    sel_tneusss0tneusss1t=f3_tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax13.plot([Tseg20211210[k,0],Tseg20211210[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax03.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0)
    elif (k%4==1):
        mkK=ax13.plot([Tseg20211210[k,0],Tseg20211210[k,1]], [0.045,0.045], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax03.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0)
    elif (k%4==2):
        mkK=ax13.plot([Tseg20211210[k,0],Tseg20211210[k,1]], [0.03,0.03], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax03.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0)
    elif (k%4==3):
        mkK=ax13.plot([Tseg20211210[k,0],Tseg20211210[k,1]], [0.015,0.015], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax03.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0)

    sel_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],f3_tObs[selID] ))
    pos_neu0=np.array([[refN0],[refE0],[refU0]])
    f3_tNEUsss[k,0],f3_tNEUsss[k,1],f3_tNEUsss[k,2],f3_tNEUsss[k,3],f3_tNEUsss[k,4],f3_tNEUsss[k,5],f3_tNEUsss[k,6], TRes = lsq_fixC(pos_neu0,sel_tneusss0neusss1T,sigma0,hmean_acous_c0s[3],twtT_thres_rms_ratio)        
    pltHZ=ax23.errorbar(f3_tNEUsss[k,2]*100.0,f3_tNEUsss[k,1]*100.0,xerr=100.0*f3_tNEUsss[k,5],yerr=100.0*f3_tNEUsss[k,4],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    pltU =ax33.errorbar(f3_tNEUsss[k,0],f3_tNEUsss[k,3]*100.0,yerr=100.0*f3_tNEUsss[k,6],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    plt_res=ax43.plot(TRes[:,0],TRes[:,1]*100, c=colors[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
    f3_counts[k]=TRes.shape[0]

    pos_neuc0_freeC=np.array([[refN0],[refE0],[refU0],[1500.0]])
    f3_tNEUsssC_freeC[k,0],f3_tNEUsssC_freeC[k,1],f3_tNEUsssC_freeC[k,2],f3_tNEUsssC_freeC[k,3],f3_tNEUsssC_freeC[k,4],f3_tNEUsssC_freeC[k,5],f3_tNEUsssC_freeC[k,6],f3_tNEUsssC_freeC[k,7], TRes_freeC = lsq_freeC(pos_neuc0_freeC,sel_tneusss0neusss1T,sigma0,twtT_thres_rms_ratio)        
    plt_res_freeC=ax43.plot(TRes_freeC[:,0],TRes_freeC[:,1]*100, c='k', marker='o',markersize=0.5,zorder=50,figure=fig,lw=0)

f3_pltNE_freeC=ax23.plot(f3_tNEUsssC_freeC[:,2]*100.0,f3_tNEUsssC_freeC[:,1]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)
f3_pltU_freeC=ax33.plot(f3_tNEUsssC_freeC[:,0],f3_tNEUsssC_freeC[:,3]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)



#For test on 2023-03-21
f4_tNEUsssC_freeC=np.zeros([Tseg20220321.shape[0],8])
f4_tNEUsss=np.zeros([Tseg20220321.shape[0],7])
f4_counts=np.zeros(Tseg20220321.shape[0])

for k in range(0,len(Tseg20220321)):
    selID=np.where( (f4_tObs>=Tseg20220321[k,0]) & (f4_tObs<Tseg20220321[k,1]) & (np.sqrt(f4_tneusss0tneusss1t[:,1]**2 + f4_tneusss0tneusss1t[:,2]**2)<=120.0) )[0]

    sel_tneusss0tneusss1t=f4_tneusss0tneusss1t[selID,:]
    if (k%4==0):
        mkK=ax14.plot([Tseg20220321[k,0],Tseg20220321[k,1]], [0.06,0.06], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax04.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='o',markersize=0.3,zorder=50,figure=fig,lw=0)
    elif (k%4==1):
        mkK=ax14.plot([Tseg20220321[k,0],Tseg20220321[k,1]], [0.045,0.045], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax04.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='+',markersize=2.5,zorder=45,figure=fig,lw=0)
    elif (k%4==2):
        mkK=ax14.plot([Tseg20220321[k,0],Tseg20220321[k,1]], [0.03,0.03], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax04.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='^',markersize=2,zorder=40,figure=fig,lw=0)
    elif (k%4==3):
        mkK=ax14.plot([Tseg20220321[k,0],Tseg20220321[k,1]], [0.015,0.015], c=colors[k], zorder=15,figure=fig,lw=1)
        pltK=ax04.plot(sel_tneusss0tneusss1t[:,2],sel_tneusss0tneusss1t[:,1], c=colors[k], marker='s',markersize=2,zorder=35,figure=fig,lw=0)

    sel_tneusss0neusss1T=np.column_stack(( sel_tneusss0tneusss1t[:,14],sel_tneusss0tneusss1t[:,1:7],sel_tneusss0tneusss1t[:,8:14],f4_tObs[selID] ))
    pos_neu0=np.array([[refN0],[refE0],[refU0]])
    f4_tNEUsss[k,0],f4_tNEUsss[k,1],f4_tNEUsss[k,2],f4_tNEUsss[k,3],f4_tNEUsss[k,4],f4_tNEUsss[k,5],f4_tNEUsss[k,6], TRes = lsq_fixC(pos_neu0,sel_tneusss0neusss1T,sigma0,hmean_acous_c0s[4],twtT_thres_rms_ratio)        
    pltHZ=ax24.errorbar(f4_tNEUsss[k,2]*100.0,f4_tNEUsss[k,1]*100.0,xerr=100.0*f4_tNEUsss[k,5],yerr=100.0*f4_tNEUsss[k,4],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    pltU =ax34.errorbar(f4_tNEUsss[k,0],f4_tNEUsss[k,3]*100.0,yerr=100.0*f4_tNEUsss[k,6],fmt='o',c=colors[k],markersize=2,lw=0.5,capsize=2,capthick=0.5,figure=fig)
    plt_res=ax44.plot(TRes[:,0],TRes[:,1]*100, c=colors[k], marker='o',markersize=0.5,zorder=5,figure=fig,lw=0)
    f4_counts[k]=TRes.shape[0]

    pos_neuc0_freeC=np.array([[refN0],[refE0],[refU0],[1500.0]])
    f4_tNEUsssC_freeC[k,0],f4_tNEUsssC_freeC[k,1],f4_tNEUsssC_freeC[k,2],f4_tNEUsssC_freeC[k,3],f4_tNEUsssC_freeC[k,4],f4_tNEUsssC_freeC[k,5],f4_tNEUsssC_freeC[k,6],f4_tNEUsssC_freeC[k,7], TRes_freeC = lsq_freeC(pos_neuc0_freeC,sel_tneusss0neusss1T,sigma0,twtT_thres_rms_ratio)        
    plt_res_freeC=ax44.plot(TRes_freeC[:,0],TRes_freeC[:,1]*100, c='k', marker='o',markersize=0.5,zorder=50,figure=fig,lw=0)

f4_pltNE_freeC=ax24.plot(f4_tNEUsssC_freeC[:,2]*100.0,f4_tNEUsssC_freeC[:,1]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)
f4_pltU_freeC=ax34.plot(f4_tNEUsssC_freeC[:,0],f4_tNEUsssC_freeC[:,3]*100.0, c='k', marker='o',markersize=1,figure=fig,lw=0,zorder=50)




fig.savefig('Figure9.pdf',dpi=300)
plt.show()

