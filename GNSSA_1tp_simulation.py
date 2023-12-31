"""
Copyright (c) 2023, Surui Xie
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 



This program is used to simulate a single transponder GNSS-A in shallow water, tested with Python 3.9.5 in macOS v11.6.6

Readers should be aware that some of the settings were used to intentionally produce exactly the same figure as Figure 3 in the published paper.

Example run:
    python GNSSA_1tp_simulation.py -z data/ray_tracing_z.npy -r data/ray_tracing_r.npy -tt data/ray_tracing_travel_time.npy

Reference: Xie, S., Zumberge, M., Sasagawa, G., and Voytenko, D., 202x. Shallow Water Seafloor Geodesy with Wave Glider-Based GNSS-Acoustic Surveying of a Single Transponder. Submitted to Earth and Space Science
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import sparse, interpolate
from gekko import GEKKO
import sys

##################
# Below we define some inputs
##################
htransd=0.38            #transducer's depth below sea surface
acous_interval=10.0     #acoustic ranging interval, unit: second
depth=54.0              #transponder depth below trasducer, unit: meter
radius=120.0            #circle drive radius, unit: meter

hmean_acous_c0=1505.120   #harmonic mean velocity between the transducer and the seafloor, unit: m/s
bias_acous_c0=5.0         #simulated bias in harmonic mean sound speed, unit: m/s
bias_acous_c0_smaller=1.0       #a smaller bias in the harmonic mean sound speed used in the simulation, for comparison purpose, unit: m/s
rate_hmean_acous_c0=1.0/3600.0  #simulated sound speed change, unit: m/s per second

tat=0.2      #transponder turn-around time in second
WG_v0=1.0    #Wave Glider speed between the times of signal transmitting and receiving (m/s)

biasN=5.0    #Northern component of the offset between the transponder and the circle-drive center, unit: meter
biasE=-5.0   #Eastern component of the offset between the transponder and the circle-drive center, unit: meter

refN0=0    #A priori transponder coordinate in local topocentric system (North), unit: meter
refE0=0    #A priori transponder coordinate in local topocentric system (East), unit: meter
refU0=0    #A priori transponder coordinate in local topocentric system (Up), unit: meter

#Transducer position precision used for the simulation
sigGNSS_NE=0.015    #unit: m
sigGNSS_U=0.045     #unit: m
sigma0=np.sqrt(2*sigGNSS_NE*sigGNSS_NE + sigGNSS_U**2) #a priori 1-sigma range error (unit: m)


"""
Below is the simulated white noise added to the transducer position, generated with numpy.random.normal.
The reason to input this file is to keep the figure consistent with the published paper. Random noise generated by different runs of numpy.random.normal will produce slight different results.
"""
nobs0=76
f_wn='data/NEUnoise_simulated_ESS.txt'
sss=np.loadtxt(f_wn)

f_ray_tracing_z = sys.argv[sys.argv.index("-z")+1]
f_ray_tracing_r = sys.argv[sys.argv.index("-r")+1]
f_ray_tracing_travel_time = sys.argv[sys.argv.index("-tt")+1]

ray_tracing_z=np.load(f_ray_tracing_z)
ray_tracing_r=np.load(f_ray_tracing_r)
ray_tracing_travel_time=np.load(f_ray_tracing_travel_time)

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

def optHZ_ID(A0, W0, l0):
    number_data=len(l0)
    Wdiag0=W0.diagonal()
    N11_sub0=(A0[:,0]**2)*Wdiag0
    N22_sub0=(A0[:,1]**2)*Wdiag0
    N33_sub0=(A0[:,2]**2)*Wdiag0
    N12_sub0=(A0[:,0]*A0[:,1])*Wdiag0
    N13_sub0=(A0[:,0]*A0[:,2])*Wdiag0
    N23_sub0=(A0[:,1]*A0[:,2])*Wdiag0
    AtWl1_sub0=A0[:,0]*Wdiag0*l0
    AtWl2_sub0=A0[:,1]*Wdiag0*l0
    AtWl3_sub0=A0[:,2]*Wdiag0*l0 
    
    """
    #Optimize the observation geometry
    #GEKKO software is used for optimization
    #GEKKO website: https://gekko.readthedocs.io/en/latest/index.html#
    """
    
    om = GEKKO(remote=False) # Initialize the optimization model
    om.options.SOLVER=1  # APOPT solver: 1
    om.solver_options = ['minlp_maximum_iterations 1000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 1000', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 2000', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.001', \
                        # covergence tolerance
                        'minlp_gap_tol 0.001']
    data=np.column_stack((N11_sub0,N22_sub0,N33_sub0,N12_sub0,N13_sub0,N23_sub0,AtWl1_sub0,AtWl2_sub0,AtWl3_sub0,np.ones(number_data)))
    data=np.around(data,decimals=5)

    ob=om.Array(om.Var,(data.shape[0],1),value=1,lb=0,ub=1,integer=True)
    nn=((data*ob).sum(axis=0))
    
    Ndiags=np.array([nn[0], nn[1], nn[2]])
    Ndiag=om.Intermediate(om.sum(Ndiags))    
    
    non_Ndiags=np.array([om.abs(nn[3]), om.abs(nn[4]), om.abs(nn[5])])
    non_diag=om.Intermediate(om.sum(non_Ndiags))    
        
    om.Equation(nn[9]> number_data*0.6667 )
    om.Minimize(non_diag/Ndiag)
    
    print ('Optimizing observation geometry for single transponder GNSS-A ...')
    om.solve(debug=0,disp=True)
    lob=np.array([item[0][0] for item in ob])
    lob=lob.reshape(len(lob),1)
    sID=np.where(lob.flatten()==1)[0]
    return sID


#Initiate a figure to show sea surface trajectory and seafloor transponder position estimates
#Settings below are rather arbitrary.
fig = plt.figure(figsize=(6.5,6.6))
gs = gridspec.GridSpec(3, 4, height_ratios=[1,1,1], width_ratios=[1,1,0.2,0.7])
gs.update(hspace=0.03,wspace=0.05)
ax00 = fig.add_subplot(gs[0,0])
ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
ax01 = fig.add_subplot(gs[0,1], sharey=ax00)
ax11 = fig.add_subplot(gs[1,1], sharex=ax01, sharey=ax10)
ax21 = fig.add_subplot(gs[2,1], sharex=ax01, sharey=ax20)

ax02 = fig.add_subplot(gs[0,3])
ax12 = fig.add_subplot(gs[1,3])
ax22 = fig.add_subplot(gs[2,3])
fig.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.99)


for ax in [ax00,ax10,ax20,ax01,ax11,ax21]:
    ax.set_aspect('equal')
    ax.set_xlim(-130,130)
    ax.set_ylim(-130,130)
    ax.set_xticks([-100,0,100])
    ax.set_yticks([-100,0,100])
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls='--')
    if ax in [ax00,ax10,ax20]:
        ax.set_ylabel("North (m)",labelpad=2)
    if ax in [ax20,ax21]:
        ax.set_xlabel("East (m)",labelpad=2)
    plt_transp=ax.plot(0,0, c='grey', marker='H',markersize=8, mew=0,zorder=10,figure=fig,lw=0)

for ax in [ax02,ax12,ax22]:
    ax.set_aspect('equal')
    ax.set_xlim(-4.5,4.5)
    ax.set_ylim(-4.5,4.5)
    ax.set_xticks([-4,-2,0,2,4])
    ax.set_yticks([-4,-2,0,2,4])
    ax.set_xlabel("East (cm)",labelpad=2)
    ax.set_ylabel("North (cm)",labelpad=2)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, va="center")
    ax.grid(True,c='lightgrey',zorder=0,lw=0.5,ls='--')

for ax in [ax01,ax11,ax21]:
    ax.tick_params(labelleft=False)
for ax in [ax00,ax01,ax10,ax11]:
    ax.tick_params(labelbottom=False)

ax00.text(-130,130, "a", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax01.text(-130,130, "b", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax02.text(-4.5,4.5, "c", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

ax10.text(-130,130, "d", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax11.text(-130,130, "e", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax12.text(-4.5,4.5, "f", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)

ax20.text(-130,130, "g", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax21.text(-130,130, "h", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)
ax22.text(-4.5,4.5, "i", color='k', fontweight='bold',fontsize=15,ha='left',va='top', clip_on=False, zorder = 5, figure=fig)


"""
Plot observations in Scenario 00. Observations evenly distributed around a circle above the transponder, transponder located below the circle
"""
angles_rad00=np.linspace(0,2.0*np.pi,nobs0,endpoint=False)
surface_e00=radius*np.sin(angles_rad00)
surface_n00=radius*np.cos(angles_rad00)
plt_surface00=ax00.plot(surface_e00,surface_n00, c='k', marker='o',markersize=2.5,zorder=5,figure=fig,lw=0)

"""
Plot observations in Scenario 10. Observations evenly distributed around a circle above the transponder, the center has shifted by some distances from the transponder
"""
surface_e10=surface_e00+biasE
surface_n10=surface_n00+biasN
plt_surface10=ax10.plot(surface_e10,surface_n10, c='k', marker='o',markersize=2.5,zorder=5,figure=fig,lw=0)


"""
Plot observations in Scenario 20. Observations are un-evenly distributed around a circle above the transponder
"""
angles_rad20a=np.linspace(0,1.5*np.pi,int(angles_rad00.shape[0]*2/3))
incr_angle_rad20b= 0.5*np.pi/ (angles_rad00.shape[0]-angles_rad20a.shape[0]+1)
angles_rad20b=np.arange(1.5*np.pi+incr_angle_rad20b,2.0*np.pi-0.1*incr_angle_rad20b,incr_angle_rad20b)
angles_rad20=np.concatenate((angles_rad20a,angles_rad20b))
surface_e20=radius*np.sin(angles_rad20)
surface_n20=radius*np.cos(angles_rad20)
plt_surface20=ax20.plot(surface_e20,surface_n20, c='k', marker='o',markersize=2.5,zorder=5,figure=fig,lw=0)

"""
Plot observations in Scenario 01. Observations are evenly distributed around a circle above the transponder, but the acoustic speed changes over time
"""
plt_surface01=ax01.scatter(surface_e00,surface_n00, c=range(0,nobs0), s=12, edgecolor='none',linewidths=0.0, marker='s', cmap='jet', vmin=0, vmax=nobs0, zorder=5, figure=fig)

"""
Plot observations in Scenario 11.
"""
plt_surface11=ax11.scatter(surface_e10,surface_n10, c=range(0,nobs0), s=12, edgecolor='none',linewidths=0.0, marker='s', cmap='jet', vmin=0, vmax=nobs0, zorder=5, figure=fig)

"""
Plot observations in Scenario 21.
"""
plt_surface21=ax21.scatter(surface_e20,surface_n20, c=range(0,nobs0), s=12, edgecolor='none',linewidths=0.0, marker='s', cmap='jet', vmin=0, vmax=nobs0, zorder=5, figure=fig)

"""
For each scenario
1. Forward predicting the two-way travel time based on acoustic ray tracing
2. Estimate the transponder position based on simulated data.
"""
id_ray_z = np.where( ( ray_tracing_z>=(depth-3*(ray_tracing_z[1]-ray_tracing_z[0])) ) & ( ray_tracing_z<=(depth+3*(ray_tracing_z[1]-ray_tracing_z[0])) ) )[0]
ray_zs=ray_tracing_z[id_ray_z]
ray_1way_travel_times=ray_tracing_travel_time[id_ray_z,:]
fint_rt_t=interpolate.RectBivariateSpline(ray_zs,ray_tracing_r,ray_1way_travel_times)

epoch_Ts = acous_interval * np.arange(0,nobs0,1.0)  #Observation time in seconds since the first ping.

"""
Predicting the two-way travel time for Scenario 00
##white noise is added to the "true" positions, assuming at GNSS positioning error level
"""
S00_tneusss0neusss1 = np.zeros([nobs0,13])
n00a=surface_n00
e00a=surface_e00
range00a=np.sqrt( (n00a-refN0)**2 + (e00a-refE0)**2 )
tt00a=fint_rt_t.ev(depth,range00a)

fn00b=interpolate.interp1d(epoch_Ts,n00a, kind='linear', fill_value='extrapolate')
n00b=fn00b(epoch_Ts+tat+2.0*tt00a)
fe00b=interpolate.interp1d(epoch_Ts,e00a, kind='linear', fill_value='extrapolate')
e00b=fe00b(epoch_Ts+tat+2.0*tt00a )
dist00b=np.sqrt( (n00b-refN0)**2 + (e00b-refE0)**2 )
tt00b=fint_rt_t.ev(depth,dist00b)

S00_tneusss0neusss1[:,0]=tt00a+tt00b
S00_tneusss0neusss1[:,1]=n00a+sss[0:nobs0,0]
S00_tneusss0neusss1[:,2]=e00a+sss[0:nobs0,1]
S00_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S00_tneusss0neusss1[:,4:6]=sigGNSS_NE
S00_tneusss0neusss1[:,6]=sigGNSS_U
S00_tneusss0neusss1[:,7]= n00b + sss[nobs0:(2*nobs0),0]
S00_tneusss0neusss1[:,8]= e00b + sss[nobs0:(2*nobs0),1]
S00_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S00_tneusss0neusss1[:,10:12]=sigGNSS_NE
S00_tneusss0neusss1[:,12]=sigGNSS_U

S00_A0, S00_W0, S00_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S00_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S00_AtW0=sparse.csr_matrix.dot(S00_A0.T,S00_W0)
S00_AtWA0=np.dot(S00_AtW0,S00_A0)
S00_AtWl0=np.dot(S00_AtW0,S00_l0)
S00_est0=np.linalg.solve( S00_AtWA0, S00_AtWl0 )
print ('Scenario 00: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S00_est0[0,0],S00_est0[1,0],S00_est0[2,0]))
pltS00_est0=ax02.plot(S00_est0[1,0]*100,S00_est0[0,0]*100, c='k', marker='o',markersize=4,zorder=5,figure=fig,lw=0)

S00_N1_inv=np.linalg.inv(S00_AtWA0)
S00_hdop=np.sqrt( S00_N1_inv[0,0] + S00_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S00_hdop)


S00_sID = optHZ_ID(S00_A0, S00_W0, S00_tneusss0neusss1[:,0])
plt_surface00_sel=ax00.plot(S00_tneusss0neusss1[S00_sID,2],S00_tneusss0neusss1[S00_sID,1], c='r', marker='+',markersize=5,mew=0.6,zorder=6,figure=fig,lw=0)

S00_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S00_A1, S00_W1, S00_l1 = getAWl(S00_est1_neu1,S00_tneusss0neusss1[S00_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S00_AtW1=sparse.csr_matrix.dot(S00_A1.T,S00_W1)
S00_AtWA1=np.dot(S00_AtW1,S00_A1)
S00_AtWl1=np.dot(S00_AtW1,S00_l1)
S00_d_neu=np.linalg.solve( S00_AtWA1, S00_AtWl1 )
kcount=0
while ( ( ( np.abs(S00_d_neu[0,0]) >1.0e-7) or (np.abs(S00_d_neu[1,0]) >1.0e-7) or (np.abs(S00_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S00_est1_neu1=S00_est1_neu1+S00_d_neu
    S00_A1, S00_W1, S00_l1 = getAWl(S00_est1_neu1,S00_tneusss0neusss1[S00_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S00_AtW1=sparse.csr_matrix.dot(S00_A1.T,S00_W1)
    S00_AtWA1=np.dot(S00_AtW1,S00_A1)
    S00_AtWl1=np.dot(S00_AtW1,S00_l1)
    S00_d_neu=np.linalg.solve( S00_AtWA1, S00_AtWl1 )
    kcount=kcount+1 

print ('Scenario 00: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S00_est1_neu1[0,0],S00_est1_neu1[1,0],S00_est1_neu1[2,0]))
pltS00_est1=ax02.plot(S00_est1_neu1[1,0]*100,S00_est1_neu1[0,0]*100, c='r', marker='+',markersize=6, mew=1,zorder=6,figure=fig,lw=0)

S00_N1_inv=np.linalg.inv(S00_AtWA1)
S00_hdop1=np.sqrt( S00_N1_inv[0,0] + S00_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S00_hdop1)




"""
Predicting the two-way travel time for Scenario 10
"""
n10a=surface_n10
e10a=surface_e10
dist10a=np.sqrt( (n10a-refN0)**2 + (e10a-refE0)**2 )
tt10a=fint_rt_t.ev(depth,dist10a)

fn10b=interpolate.interp1d(epoch_Ts,n10a, kind='linear', fill_value='extrapolate')
n10b=fn10b(epoch_Ts+tat+2.0*tt10a)
fe10b=interpolate.interp1d(epoch_Ts,e10a, kind='linear', fill_value='extrapolate')
e10b=fe10b(epoch_Ts+tat+2.0*tt10a )
dist10b=np.sqrt( (n10b-refN0)**2 + (e10b-refE0)**2 )
tt10b=fint_rt_t.ev(depth,dist10b)

S10_tneusss0neusss1 = np.zeros([nobs0,13])
S10_tneusss0neusss1[:,0]=tt10a+tt10b
S10_tneusss0neusss1[:,1]=n10a+sss[0:nobs0,0]
S10_tneusss0neusss1[:,2]=e10a+sss[0:nobs0,1]
S10_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S10_tneusss0neusss1[:,4:6]=sigGNSS_NE
S10_tneusss0neusss1[:,6]=sigGNSS_U
S10_tneusss0neusss1[:,7]= n10b + sss[nobs0:(2*nobs0),0]
S10_tneusss0neusss1[:,8]= e10b + sss[nobs0:(2*nobs0),1]
S10_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S10_tneusss0neusss1[:,10:12]=sigGNSS_NE
S10_tneusss0neusss1[:,12]=sigGNSS_U

S10_A0, S10_W0, S10_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S10_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S10_AtW0=sparse.csr_matrix.dot(S10_A0.T,S10_W0)
S10_AtWA0=np.dot(S10_AtW0,S10_A0)
S10_AtWl0=np.dot(S10_AtW0,S10_l0)
S10_est0=np.linalg.solve( S10_AtWA0, S10_AtWl0 )
print ('Scenario 10: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S10_est0[0,0],S10_est0[1,0],S10_est0[2,0]))
pltS10_est0=ax12.plot(S10_est0[1,0]*100,S10_est0[0,0]*100, c='k', marker='o',markersize=4,zorder=5,figure=fig,lw=0)
S10_N1_inv=np.linalg.inv(S10_AtWA0)
S10_hdop=np.sqrt( S10_N1_inv[0,0] + S10_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S10_hdop)

S10_sID = optHZ_ID(S10_A0, S10_W0, S10_tneusss0neusss1[:,0])
plt_surface10_sel=ax10.plot(S10_tneusss0neusss1[S10_sID,2],S10_tneusss0neusss1[S10_sID,1], c='r', marker='+',markersize=5,mew=0.6,zorder=6,figure=fig,lw=0)


S10_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S10_A1, S10_W1, S10_l1 = getAWl(S10_est1_neu1,S10_tneusss0neusss1[S10_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S10_AtW1=sparse.csr_matrix.dot(S10_A1.T,S10_W1)
S10_AtWA1=np.dot(S10_AtW1,S10_A1)
S10_AtWl1=np.dot(S10_AtW1,S10_l1)
S10_d_neu=np.linalg.solve( S10_AtWA1, S10_AtWl1 )
kcount=0
while ( ( ( np.abs(S10_d_neu[0,0]) >1.0e-7) or (np.abs(S10_d_neu[1,0]) >1.0e-7) or (np.abs(S10_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S10_est1_neu1=S10_est1_neu1+S10_d_neu
    S10_A1, S10_W1, S10_l1 = getAWl(S10_est1_neu1,S10_tneusss0neusss1[S10_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S10_AtW1=sparse.csr_matrix.dot(S10_A1.T,S10_W1)
    S10_AtWA1=np.dot(S10_AtW1,S10_A1)
    S10_AtWl1=np.dot(S10_AtW1,S10_l1)
    S10_d_neu=np.linalg.solve( S10_AtWA1, S10_AtWl1 )
    kcount=kcount+1 

print ('Scenario 10: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S10_est1_neu1[0,0],S10_est1_neu1[1,0],S10_est1_neu1[2,0]))
pltS10_est1=ax12.plot(S10_est1_neu1[1,0]*100,S10_est1_neu1[0,0]*100, c='r', marker='+',markersize=6, mew=1,zorder=6,figure=fig,lw=0)
S10_N1_inv=np.linalg.inv(S10_AtWA1)
S10_hdop1=np.sqrt( S10_N1_inv[0,0] + S10_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S10_hdop1)


####***** smaller sound speed error, 1 m/s
S10_A0, S10_W0, S10_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S10_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0_smaller)
S10_AtW0=sparse.csr_matrix.dot(S10_A0.T,S10_W0)
S10_AtWA0=np.dot(S10_AtW0,S10_A0)
S10_AtWl0=np.dot(S10_AtW0,S10_l0)
S10_est0=np.linalg.solve( S10_AtWA0, S10_AtWl0 )
pltS10_est0=ax12.plot(S10_est0[1,0]*100,S10_est0[0,0]*100, c='none', marker='o',mew=1,mec='k',markersize=4,zorder=5,figure=fig,lw=0)

"""
Predicting the two-way travel time for Scenario 20
"""
n20a=surface_n20
e20a=surface_e20
dist20a=np.sqrt( (n20a-refN0)**2 + (e20a-refE0)**2 )
tt20a=fint_rt_t.ev(depth,dist20a)

fn20b=interpolate.interp1d(epoch_Ts,n20a, kind='linear', fill_value='extrapolate')
n20b=fn20b(epoch_Ts+tat+2.0*tt20a)
fe20b=interpolate.interp1d(epoch_Ts,e20a, kind='linear', fill_value='extrapolate')
e20b=fe20b(epoch_Ts+tat+2.0*tt20a )
dist20b=np.sqrt( (n20b-refN0)**2 + (e20b-refE0)**2 )
tt20b=fint_rt_t.ev(depth,dist20b)

S20_tneusss0neusss1 = np.zeros([nobs0,13])
S20_tneusss0neusss1[:,0]=tt20a+tt20b
S20_tneusss0neusss1[:,1]=n20a+sss[0:nobs0,0]
S20_tneusss0neusss1[:,2]=e20a+sss[0:nobs0,1]
S20_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S20_tneusss0neusss1[:,4:6]=sigGNSS_NE
S20_tneusss0neusss1[:,6]=sigGNSS_U
S20_tneusss0neusss1[:,7]= n20b + sss[nobs0:(2*nobs0),0]
S20_tneusss0neusss1[:,8]= e20b + sss[nobs0:(2*nobs0),1]
S20_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S20_tneusss0neusss1[:,10:12]=sigGNSS_NE
S20_tneusss0neusss1[:,12]=sigGNSS_U

S20_A0, S20_W0, S20_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S20_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S20_AtW0=sparse.csr_matrix.dot(S20_A0.T,S20_W0)
S20_AtWA0=np.dot(S20_AtW0,S20_A0)
S20_AtWl0=np.dot(S20_AtW0,S20_l0)
S20_est0=np.linalg.solve( S20_AtWA0, S20_AtWl0 )
print ('Scenario 20: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S20_est0[0,0],S20_est0[1,0],S20_est0[2,0]))
pltS20_est0=ax22.plot(S20_est0[1,0]*100,S20_est0[0,0]*100, c='k', marker='o',markersize=4,zorder=5,figure=fig,lw=0)
S20_N1_inv=np.linalg.inv(S20_AtWA0)
S20_hdop=np.sqrt( S20_N1_inv[0,0] + S20_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S20_hdop)


S20_sID = optHZ_ID(S20_A0, S20_W0, S20_tneusss0neusss1[:,0])
plt_surface10_sel=ax20.plot(S20_tneusss0neusss1[S20_sID,2],S20_tneusss0neusss1[S20_sID,1], c='r', marker='+',markersize=5,mew=0.6,zorder=6,figure=fig,lw=0)


S20_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S20_A1, S20_W1, S20_l1 = getAWl(S20_est1_neu1,S20_tneusss0neusss1[S20_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S20_AtW1=sparse.csr_matrix.dot(S20_A1.T,S20_W1)
S20_AtWA1=np.dot(S20_AtW1,S20_A1)
S20_AtWl1=np.dot(S20_AtW1,S20_l1)
S20_d_neu=np.linalg.solve( S20_AtWA1, S20_AtWl1 )
kcount=0
while ( ( ( np.abs(S20_d_neu[0,0]) >1.0e-7) or (np.abs(S20_d_neu[1,0]) >1.0e-7) or (np.abs(S20_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S20_est1_neu1=S20_est1_neu1+S20_d_neu
    S20_A1, S20_W1, S20_l1 = getAWl(S20_est1_neu1,S20_tneusss0neusss1[S20_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S20_AtW1=sparse.csr_matrix.dot(S20_A1.T,S20_W1)
    S20_AtWA1=np.dot(S20_AtW1,S20_A1)
    S20_AtWl1=np.dot(S20_AtW1,S20_l1)
    S20_d_neu=np.linalg.solve( S20_AtWA1, S20_AtWl1 )
    kcount=kcount+1 

print ('Scenario 20: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S20_est1_neu1[0,0],S20_est1_neu1[1,0],S20_est1_neu1[2,0]))
pltS20_est1=ax22.plot(S20_est1_neu1[1,0]*100,S20_est1_neu1[0,0]*100, c='r', marker='+',markersize=6, mew=1,zorder=6,figure=fig,lw=0)

S20_N1_inv=np.linalg.inv(S20_AtWA1)
S20_hdop1=np.sqrt( S20_N1_inv[0,0] + S20_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S20_hdop1)


"""
Predicting the two-way travel time for Scenario 01
"""

n01a=surface_n00
e01a=surface_e00

n01b=n00b
e01b=e00b

dist01a=np.sqrt( (n01a-refN0)**2 + (e01a-refE0)**2 + (depth-htransd)**2 )
dist01b=np.sqrt( (n01b-refN0)**2 + (e01b-refE0)**2 + (depth-htransd)**2 )


intercept_fit_acousC=hmean_acous_c0-rate_hmean_acous_c0*np.mean(epoch_Ts)
tt01a = dist01a/(rate_hmean_acous_c0*epoch_Ts+intercept_fit_acousC)
tt01b = dist01a/(rate_hmean_acous_c0*(epoch_Ts+tat+tt01a)+intercept_fit_acousC)


S01_tneusss0neusss1 = np.zeros([nobs0,13])
S01_tneusss0neusss1[:,0]=tt01a+tt01b
S01_tneusss0neusss1[:,1]=n01a+sss[0:nobs0,0]
S01_tneusss0neusss1[:,2]=e01a+sss[0:nobs0,1]
S01_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S01_tneusss0neusss1[:,4:6]=sigGNSS_NE
S01_tneusss0neusss1[:,6]=sigGNSS_U
S01_tneusss0neusss1[:,7]= n01b + sss[nobs0:(2*nobs0),0]
S01_tneusss0neusss1[:,8]= e01b + sss[nobs0:(2*nobs0),1]
S01_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S01_tneusss0neusss1[:,10:12]=sigGNSS_NE
S01_tneusss0neusss1[:,12]=sigGNSS_U

S01_A0, S01_W0, S01_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S01_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S01_AtW0=sparse.csr_matrix.dot(S01_A0.T,S01_W0)
S01_AtWA0=np.dot(S01_AtW0,S01_A0)
S01_AtWl0=np.dot(S01_AtW0,S01_l0)
S01_est0=np.linalg.solve( S01_AtWA0, S01_AtWl0 )
print ('Scenario 01: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S01_est0[0,0],S01_est0[1,0],S01_est0[2,0]))
pltS01_est0=ax02.plot(S01_est0[1,0]*100,S01_est0[0,0]*100, c='b', marker='s',markersize=4,zorder=5,figure=fig,lw=0)
S01_N1_inv=np.linalg.inv(S01_AtWA0)
S01_hdop=np.sqrt( S01_N1_inv[0,0] + S01_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S01_hdop)

for S01_angle in [0.5*np.pi,np.pi,1.5*np.pi]:
    S01_e=S01_est0[1,0]*np.cos(S01_angle) - S01_est0[0,0]*np.sin(S01_angle)
    S01_n=S01_est0[1,0]*np.sin(S01_angle) + S01_est0[0,0]*np.cos(S01_angle)
    pltS01_angle=ax02.plot(S01_e*100,S01_n*100, c='lightblue', marker='s',markersize=4,zorder=5,figure=fig,lw=0)


for k in range(1,4):
    tmp_angle=0.5*np.pi*k
    tmp_radius=radius-k*20
    angles_rad=np.linspace(tmp_angle,1.93*np.pi+tmp_angle-k*0.015*np.pi,100)
    #angles_rad=np.arange(tmp_angle,1.9*np.pi+tmp_angle,0.01*np.pi)
    etmp=tmp_radius*np.sin(angles_rad)
    ntmp=tmp_radius*np.cos(angles_rad)
    plt_start=ax01.plot(etmp[0],ntmp[0], c='lightblue', marker='s',markersize=4,zorder=5,figure=fig,lw=0)
    plt_tmp=ax01.plot(etmp,ntmp, c='lightblue', zorder=15,figure=fig,lw=1)
    plt_vector=ax01.arrow(etmp[-1],ntmp[-1],etmp[-1]-etmp[-2],ntmp[-1]-ntmp[-2], color='lightblue',head_width=5,head_length=8, lw=1,zorder=15, figure=fig)


S01_sID = optHZ_ID(S01_A0, S01_W0, S01_tneusss0neusss1[:,0])
plt_surface10_sel=ax01.plot(S01_tneusss0neusss1[S01_sID,2],S01_tneusss0neusss1[S01_sID,1], c='r', marker='x',markersize=5,mew=0.5,zorder=6,figure=fig,lw=0)
S01_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S01_A1, S01_W1, S01_l1 = getAWl(S01_est1_neu1,S01_tneusss0neusss1[S01_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S01_AtW1=sparse.csr_matrix.dot(S01_A1.T,S01_W1)
S01_AtWA1=np.dot(S01_AtW1,S01_A1)
S01_AtWl1=np.dot(S01_AtW1,S01_l1)
S01_d_neu=np.linalg.solve( S01_AtWA1, S01_AtWl1 )
kcount=0
while ( ( ( np.abs(S01_d_neu[0,0]) >1.0e-7) or (np.abs(S01_d_neu[1,0]) >1.0e-7) or (np.abs(S01_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S01_est1_neu1=S01_est1_neu1+S01_d_neu
    S01_A1, S01_W1, S01_l1 = getAWl(S01_est1_neu1,S01_tneusss0neusss1[S01_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S01_AtW1=sparse.csr_matrix.dot(S01_A1.T,S01_W1)
    S01_AtWA1=np.dot(S01_AtW1,S01_A1)
    S01_AtWl1=np.dot(S01_AtW1,S01_l1)
    S01_d_neu=np.linalg.solve( S01_AtWA1, S01_AtWl1 )
    kcount=kcount+1 

print ('Scenario 01: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S01_est1_neu1[0,0],S01_est1_neu1[1,0],S01_est1_neu1[2,0]))
pltS01_est1=ax02.plot(S01_est1_neu1[1,0]*100,S01_est1_neu1[0,0]*100, c='r', marker='x',markersize=5, mew=1,zorder=6,figure=fig,lw=0)

S01_N1_inv=np.linalg.inv(S01_AtWA1)
S01_hdop1=np.sqrt( S01_N1_inv[0,0] + S01_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S01_hdop1)




######### faster changing rate in sound speed
intercept_fit_acousC=hmean_acous_c0-5*rate_hmean_acous_c0*np.mean(epoch_Ts)
tt01a = dist01a/(5*rate_hmean_acous_c0*epoch_Ts+intercept_fit_acousC)
tt01b = dist01a/(5*rate_hmean_acous_c0*(epoch_Ts+tat+tt01a)+intercept_fit_acousC)


S01_tneusss0neusss1 = np.zeros([nobs0,13])
S01_tneusss0neusss1[:,0]=tt01a+tt01b
S01_tneusss0neusss1[:,1]=n01a+sss[0:nobs0,0]
S01_tneusss0neusss1[:,2]=e01a+sss[0:nobs0,1]
S01_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S01_tneusss0neusss1[:,4:6]=sigGNSS_NE
S01_tneusss0neusss1[:,6]=sigGNSS_U
S01_tneusss0neusss1[:,7]= n01b + sss[nobs0:(2*nobs0),0]
S01_tneusss0neusss1[:,8]= e01b + sss[nobs0:(2*nobs0),1]
S01_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S01_tneusss0neusss1[:,10:12]=sigGNSS_NE
S01_tneusss0neusss1[:,12]=sigGNSS_U

S01_A0, S01_W0, S01_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S01_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S01_AtW0=sparse.csr_matrix.dot(S01_A0.T,S01_W0)
S01_AtWA0=np.dot(S01_AtW0,S01_A0)
S01_AtWl0=np.dot(S01_AtW0,S01_l0)
S01_est0=np.linalg.solve( S01_AtWA0, S01_AtWl0 )
pltS01_est0=ax02.plot(S01_est0[1,0]*100,S01_est0[0,0]*100, c='none', marker='s',mec='b',mew=1,markersize=4,zorder=5,figure=fig,lw=0)


"""
Predicting the two-way travel time for Scenario 11
"""
n11a=surface_n10
e11a=surface_e10

n11b=n10b
e11b=e10b

dist11a=np.sqrt( (n11a-refN0)**2 + (e11a-refE0)**2 + (depth-htransd)**2 )
tt11a = dist11a/(rate_hmean_acous_c0*epoch_Ts+intercept_fit_acousC)

dist11b=np.sqrt( (n11b-refN0)**2 + (e11b-refE0)**2 + (depth-htransd)**2 )
tt11b = dist11a/(rate_hmean_acous_c0*(epoch_Ts+tat+tt11a)+intercept_fit_acousC)

S11_tneusss0neusss1 = np.zeros([nobs0,13])
S11_tneusss0neusss1[:,0]=tt11a+tt11b
S11_tneusss0neusss1[:,1]=n11a+sss[0:nobs0,0]
S11_tneusss0neusss1[:,2]=e11a+sss[0:nobs0,1]
S11_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S11_tneusss0neusss1[:,4:6]=sigGNSS_NE
S11_tneusss0neusss1[:,6]=sigGNSS_U
S11_tneusss0neusss1[:,7]= n11b + sss[nobs0:(2*nobs0),0]
S11_tneusss0neusss1[:,8]= e11b + sss[nobs0:(2*nobs0),1]
S11_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S11_tneusss0neusss1[:,10:12]=sigGNSS_NE
S11_tneusss0neusss1[:,12]=sigGNSS_U

S11_A0, S11_W0, S11_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S11_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S11_AtW0=sparse.csr_matrix.dot(S11_A0.T,S11_W0)
S11_AtWA0=np.dot(S11_AtW0,S11_A0)
S11_AtWl0=np.dot(S11_AtW0,S11_l0)
S11_est0=np.linalg.solve( S11_AtWA0, S11_AtWl0 )
print ('Scenario 11: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S11_est0[0,0],S11_est0[1,0],S11_est0[2,0]))
pltS11_est0=ax12.plot(S11_est0[1,0]*100,S11_est0[0,0]*100, c='b', marker='s',markersize=4,zorder=5,figure=fig,lw=0)
S11_N1_inv=np.linalg.inv(S11_AtWA0)
S11_hdop=np.sqrt( S11_N1_inv[0,0] + S11_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S11_hdop)


S11_sID = optHZ_ID(S11_A0, S11_W0, S11_tneusss0neusss1[:,0])
plt_surface11_sel=ax11.plot(S11_tneusss0neusss1[S11_sID,2],S11_tneusss0neusss1[S11_sID,1], c='r', marker='x',markersize=5,mew=0.5,zorder=6,figure=fig,lw=0)
S11_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S11_A1, S11_W1, S11_l1 = getAWl(S11_est1_neu1,S11_tneusss0neusss1[S11_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S11_AtW1=sparse.csr_matrix.dot(S11_A1.T,S11_W1)
S11_AtWA1=np.dot(S11_AtW1,S11_A1)
S11_AtWl1=np.dot(S11_AtW1,S11_l1)
S11_d_neu=np.linalg.solve( S11_AtWA1, S11_AtWl1 )
kcount=0
while ( ( ( np.abs(S11_d_neu[0,0]) >1.0e-7) or (np.abs(S11_d_neu[1,0]) >1.0e-7) or (np.abs(S11_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S11_est1_neu1=S11_est1_neu1+S11_d_neu
    S11_A1, S11_W1, S11_l1 = getAWl(S11_est1_neu1,S11_tneusss0neusss1[S11_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S11_AtW1=sparse.csr_matrix.dot(S11_A1.T,S11_W1)
    S11_AtWA1=np.dot(S11_AtW1,S11_A1)
    S11_AtWl1=np.dot(S11_AtW1,S11_l1)
    S11_d_neu=np.linalg.solve( S11_AtWA1, S11_AtWl1 )
    kcount=kcount+1 

print ('Scenario 11: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S11_est1_neu1[0,0],S11_est1_neu1[1,0],S11_est1_neu1[2,0]))
pltS11_est1=ax12.plot(S11_est1_neu1[1,0]*100,S11_est1_neu1[0,0]*100, c='r', marker='x',markersize=5, mew=1,zorder=6,figure=fig,lw=0)

S11_N1_inv=np.linalg.inv(S11_AtWA1)
S11_hdop1=np.sqrt( S11_N1_inv[0,0] + S11_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S11_hdop1)



"""
Predicting the two-way travel time for Scenario 21
"""
n21a=surface_n20
e21a=surface_e20

n21b=n20b
e21b=e20b

dist21a=np.sqrt( (n21a-refN0)**2 + (e21a-refE0)**2 + (depth-htransd)**2 )
tt21a = dist21a/(rate_hmean_acous_c0*epoch_Ts+intercept_fit_acousC)

dist21b=np.sqrt( (n21b-refN0)**2 + (e21b-refE0)**2 + (depth-htransd)**2 )
tt21b = dist21a/(rate_hmean_acous_c0*(epoch_Ts+tat+tt21a)+intercept_fit_acousC)

S21_tneusss0neusss1 = np.zeros([nobs0,13])
S21_tneusss0neusss1[:,0]=tt21a+tt21b
S21_tneusss0neusss1[:,1]=n21a+sss[0:nobs0,0]
S21_tneusss0neusss1[:,2]=e21a+sss[0:nobs0,1]
S21_tneusss0neusss1[:,3]=depth-htransd+sss[0:nobs0,2]
S21_tneusss0neusss1[:,4:6]=sigGNSS_NE
S21_tneusss0neusss1[:,6]=sigGNSS_U
S21_tneusss0neusss1[:,7]= n21b + sss[nobs0:(2*nobs0),0]
S21_tneusss0neusss1[:,8]= e21b + sss[nobs0:(2*nobs0),1]
S21_tneusss0neusss1[:,9]= depth-htransd+sss[nobs0:(2*nobs0),2]
S21_tneusss0neusss1[:,10:12]=sigGNSS_NE
S21_tneusss0neusss1[:,12]=sigGNSS_U

S21_A0, S21_W0, S21_l0 = getAWl(np.array([[refN0],[refE0],[refU0]]),S21_tneusss0neusss1,sigma0,hmean_acous_c0+bias_acous_c0)
S21_AtW0=sparse.csr_matrix.dot(S21_A0.T,S21_W0)
S21_AtWA0=np.dot(S21_AtW0,S21_A0)
S21_AtWl0=np.dot(S21_AtW0,S21_l0)
S21_est0=np.linalg.solve( S21_AtWA0, S21_AtWl0 )
print ('Scenario 21: before optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S21_est0[0,0],S21_est0[1,0],S21_est0[2,0]))
pltS21_est0=ax22.plot(S21_est0[1,0]*100,S21_est0[0,0]*100, c='b', marker='s',markersize=4,zorder=5,figure=fig,lw=0)
S21_N1_inv=np.linalg.inv(S21_AtWA0)
S21_hdop=np.sqrt( S21_N1_inv[0,0] + S21_N1_inv[1,1] )/sigma0
print ('Before optimization, hdop=',S21_hdop)


S21_sID = optHZ_ID(S21_A0, S21_W0, S21_tneusss0neusss1[:,0])
plt_surface21_sel=ax21.plot(S21_tneusss0neusss1[S21_sID,2],S21_tneusss0neusss1[S21_sID,1], c='r', marker='x',markersize=5,mew=0.5,zorder=6,figure=fig,lw=0)
S21_est1_neu1=np.array([[refN0],[refE0],[refU0]])
S21_A1, S21_W1, S21_l1 = getAWl(S21_est1_neu1,S21_tneusss0neusss1[S21_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
S21_AtW1=sparse.csr_matrix.dot(S21_A1.T,S21_W1)
S21_AtWA1=np.dot(S21_AtW1,S21_A1)
S21_AtWl1=np.dot(S21_AtW1,S21_l1)
S21_d_neu=np.linalg.solve( S21_AtWA1, S21_AtWl1 )
kcount=0
while ( ( ( np.abs(S21_d_neu[0,0]) >1.0e-7) or (np.abs(S21_d_neu[1,0]) >1.0e-7) or (np.abs(S21_d_neu[2,0]) >3.0e-7) ) and (kcount <50) ):
    S21_est1_neu1=S21_est1_neu1+S21_d_neu
    S21_A1, S21_W1, S21_l1 = getAWl(S21_est1_neu1,S21_tneusss0neusss1[S21_sID,:],sigma0,hmean_acous_c0+bias_acous_c0)
    S21_AtW1=sparse.csr_matrix.dot(S21_A1.T,S21_W1)
    S21_AtWA1=np.dot(S21_AtW1,S21_A1)
    S21_AtWl1=np.dot(S21_AtW1,S21_l1)
    S21_d_neu=np.linalg.solve( S21_AtWA1, S21_AtWl1 )
    kcount=kcount+1 

print ('Scenario 21: after optimization, est_N=%.4f m, est_E=%.4f m, est_U=%.4f m' %(S21_est1_neu1[0,0],S21_est1_neu1[1,0],S21_est1_neu1[2,0]))
pltS21_est1=ax22.plot(S21_est1_neu1[1,0]*100,S21_est1_neu1[0,0]*100, c='r', marker='x',markersize=5, mew=1,zorder=6,figure=fig,lw=0)

S21_N1_inv=np.linalg.inv(S21_AtWA1)
S21_hdop1=np.sqrt( S21_N1_inv[0,0] + S21_N1_inv[1,1] )/sigma0
print ('After optimization, hdop1=',S21_hdop1)

fig.savefig('Figure3.pdf',dpi=300)
plt.show()