#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:51:35 2022

@author: holst-c
"""

# =============================================================================
# Dependencies
# =============================================================================

import sys
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import colorcet as cc

# =============================================================================
# Parameters
# =============================================================================

# Colormaps
CMAP_abs = cc.cm.CET_L11
CMAP_dif = cc.cm.CET_D1

# Plot titles
TITLE = "Test domain, PALM-4U v2204, dx = 10m, dz = 10m"

# Buffer width
BUFFER_T = 26
BUFFER_B = 16

# Vertical grid spacing
DZ = 15

# Paths
PATH_INPUT = '/Users/holst-c/Desktop/'
PATH_OUTPUT = '/Users/holst-c/Desktop/'

# Job ID
JOB_ID = 'big_suhi_default_static'

# =============================================================================
# Functions
# =============================================================================

def process_and_plot(topo_smooth: bool = True,
                     bdg_ramp:    bool = True,
                     pvmt_fix:    bool = True):
    """
    This functions adjusts static driver data and plots it.

    Some of the adjustments are to deal with bugs and inconsistencies, while
    others are aiming to allow the flow to smoothly enter the domain and avoid
    large, unrealistic velocities due to flows pushing into narrow cavities
    near the boundaries. The topography smoothing aims to enhance compatibility
    between Dirichlet forcing from larger scale models (offline nesting).
    
    Adjustments included are:

        - changing pavement_type 7 to 1
        - changing pavement_type > 15 to 1
        - smoothing boundary topography along the boundary
        - ramping building heights near the boundary
    
    """

    print("\n ... processing domain 'root' ...")

    DS1 = xr.open_dataset(f"{PATH_INPUT}{JOB_ID}_root")

    T_HGT1 = DS1["zt"].values

    D_pav1 = DS1["pavement_type"].values
    D_bui1 = DS1["building_type"].values
    D_wat1 = DS1["water_type"].values

    # D_soi = DS1["soil_type"].values
    # D_sfc = DS1["surface_fraction"].values

    D_b2d = DS1["buildings_2d"].values
    # D_b3d = DS1["buildings_3d"].values
    # D_bID = DS1["building_id"].values

    X1 = DS1["E_UTM"].values
    Y1 = DS1["N_UTM"].values

# =============================================================================
# Adjust domain 1
# =============================================================================

    # Pavement fix
    if pvmt_fix:

        S = np.where(D_pav1[:,:]==7)
        print(" ... changing {str(np.shape(S[0])[0])} entries pvmt = 7 to 1 ...")
        D_pav1[S] = 1
    
    S = np.where(D_pav1[:,:]>15)
    print(f" ... changing {str(np.shape(S[0])[0])} entries pvmt > 15 to 1 ...")
    D_pav1[S] = 1

    # Topography smoothing
    if topo_smooth:

        print(f" ... smoothing boundary topography buffer = {str(BUFFER_T)} ...")
        T_HGT1p = z_t_bdy5(T_HGT1,buffer=BUFFER_T)
        T_HGT1d = T_HGT1-T_HGT1p
        T_HGT1[:,:] = T_HGT1p

    else:

        T_HGT1p = np.copy(T_HGT1[:,:])

    # Building height ramping
    if bdg_ramp:

        print(f" ... ramping boundary buildings buffer = {str(BUFFER_B)} ...")
        D_b2d[:,:]    = b_h_bdy3(D_b2d, buffer=BUFFER_B)
        # D_b3d[:,:,:]  = b_h_bdy3_3d(D_b3d, D_b2d, buffer=BUFFER_B, DZ=DZ)

    print("\n Topography:")
    print("root min:       " + str(np.min(T_HGT1)) + " m")
    print("root max:       " + str(np.max(T_HGT1)) + " m")

    DS1.to_netcdf(f"{PATH_OUTPUT}{JOB_ID}_static")
    print(f"     saved 'root' as '{JOB_ID}_static'.")

    T_HGT1[:,:] = np.zeros_like(T_HGT1)
    DS1.to_netcdf(f"{PATH_OUTPUT}{JOB_ID}_static_flat")
    print(f"     saved 'root' as '{JOB_ID}_static_flat'.")

    DS1.close()

    print(" ... processing *_type masks ...")
    D_pav1[np.where(np.abs(D_pav1[:,:])<40.)] = 1
    D_bui1[np.where(np.abs(D_bui1[:,:])<40.)] = 1
    D_wat1[np.where(np.abs(D_wat1[:,:])<40.)] = 1

# =============================================================================
# Adjust domain 2
# =============================================================================

    try: 

        print("\n" + " ... processing domain 'N02' ...")
        DS2 = xr.open_dataset(f"{PATH_INPUT}{JOB_ID}_N02")

        T_HGT2 = DS2["zt"].values

        D_pav2 = DS2["pavement_type"].values
        D_bui2 = DS2["building_type"].values
        D_wat2 = DS2["water_type"].values

        X2 = DS2["E_UTM"].values
        Y2 = DS2["N_UTM"].values

        # Pavement fix
        if pvmt_fix:

            S = np.where(D_pav2[:,:]==7)
            print(f" ... changing {str(np.shape(S[0])[0])} entries pvmt = 7 to 1 ...")
            D_pav2[S] = 1

            S = np.where(D_pav1[:,:]>15)
            print(" ... changing {str(np.shape(S[0])[0])} entries pvmt > 15 to 1 ...")
            D_pav2[S] = 1

        DS2.to_netcdf(f"{PATH_INPUT}{JOB_ID}_static_N02")
        print(f"     saved 'N02' as '{JOB_ID}_static_N02'.")

        print("\n Topography:")
        print(f"N02  min:       {str(np.min(T_HGT2))} m")
        print(f"N02  max:       {str(np.max(T_HGT2))} m")
    
        T_HGT2p = np.copy(T_HGT2[:,:])

        T_HGT2[:,:] = np.zeros_like(T_HGT2)
        DS2.to_netcdf(f"{PATH_OUTPUT}{JOB_ID}_static_N02_flat")
        print(f"     saved 'root' as '{JOB_ID}_static_N02_flat'.")

        DS2.close()

        print(" ... processing *_type masks ...")
        D_pav2[np.where(np.abs(D_pav2[:,:])<20.)] = 1
        D_bui2[np.where(np.abs(D_bui2[:,:])<20.)] = 1
        D_wat2[np.where(np.abs(D_wat2[:,:])<20.)] = 1

        print("\n Domain details:")
        print("\n Edge of child domain:")
        print(f"N02 offset x:   {str(X2[0,0]-X1[0,0])} m")
        print(f"N02 offset y:   {str(Y2[0,0]-Y1[0,0])} m")

        print("\n Sizes:")
        print(f"root:           {str(np.shape(T_HGT1))}")
        print(f"N02:            {str(np.shape(T_HGT2))}")

    except:

        pass

    zmin = 0
    zmax = 350

# =============================================================================
# Difference plot for domain 1
# =============================================================================

    nx, ny = np.shape(T_HGT1)

    print("\n ... plotting map of masks and topo ...")

    fig = plt.figure(figsize=np.array([ny,nx])/450.*20./0.92)

    ax = fig.add_axes([.025,.025,.94,.92],frameon=True)
    
    im = ax.pcolormesh(X1, Y1, T_HGT1d, edgecolors='k', linewidth="0.4",
                  snap=True,
                  cmap=CMAP_dif,
                  zorder=1, shading="nearest",
                  vmin=-30,vmax=30)

    ax.pcolormesh(X1, Y1, D_bui1, edgecolors='none', snap=True, cmap='Reds',
                  zorder=3, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.3)

    ax.pcolormesh(X1, Y1, D_wat1, edgecolors='none', snap=True, cmap='Blues',
                  zorder=2, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.6)
    ax.set_title(TITLE)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.02)
    plt.colorbar(im, cax=cax)

    plt.savefig(f"_static_bdy_{JOB_ID}.png", dpi=600)
    print(f"     saved map as '_static_bdy_{JOB_ID}.png'.")

    plt.show()
    plt.close()

# =============================================================================
# Topo plot for domain 1
# =============================================================================

    fig = plt.figure(figsize=np.array([ny,nx])/450.*20./0.92)

    ax = fig.add_axes([.025,.025,.94,.92],frameon=True)
    
    im = ax.pcolormesh(X1, Y1, T_HGT1p, edgecolors='k', linewidth="0.1",
                  snap=True,
                  cmap=CMAP_abs,
                  zorder=1, shading="nearest",
                  vmin=zmin,vmax=zmax)

    ax.pcolormesh(X1, Y1, D_bui1, edgecolors='none', snap=True, cmap='Reds',
                  zorder=3, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.3)

    ax.pcolormesh(X1, Y1, D_wat1, edgecolors='none', snap=True, cmap='Blues',
                  zorder=2, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.6)

# =============================================================================
# Try to plot domain 2, if it exists
# =============================================================================

    try:

        im = ax.pcolormesh(X2, Y2, T_HGT2p, edgecolors='k', linewidth="0.1",
                            snap=True, shading="nearest",
                            cmap=CMAP_abs, zorder=5,
                            vmin=zmin, vmax=zmax)

        ax.pcolormesh(X2, Y2, D_bui2, edgecolors='none', snap=True, cmap='Reds',
                      zorder=7, shading="nearest",
                      vmin=0,vmax=1.05,alpha=0.3)

        ax.pcolormesh(X2, Y2, D_wat2, edgecolors='none', snap=True, cmap='Blues',
                      zorder=6, shading="nearest",
                      vmin=0,vmax=1.05,alpha=0.6)

    except:

        pass

    ax.set_title(TITLE)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.02)
    plt.colorbar(im, cax=cax)

    plt.savefig(f"_static_bdy_{JOB_ID}_N02.png", dpi=600)
    print(f"     saved map as '_static_bdy_{JOB_ID}_N02.png'.")

    plt.show()
    plt.close()

    return



def b_h_bdy3(BH, BUFFER=10):
    """
    Reduce building height near boundary linearly with closeness to boundary.

    Parameters
    ----------
    BH : numpy.array
        Building height in m.
    BUFFER : int, optional
        Buffer width, i.e. number of grid points from the edge.
        The default is 10.

    Returns
    -------
    BH2 : numpy.array
        Modified building height in m.
    
    Procedure
    ---------
    For each grid point in the boundary zone with width BUFFER:
        - Calculate mininum distance to boundary
        - Calculate weight based on distance
        - Adjust height according to weight

    """

    # Create a memory copy of the array
    BH2 = np.copy(BH)

    # Process each point inside the domain
    for i in range(BH.shape[0]):

        for j in range(BH.shape[1]):

            # Skip grid points that are outside bonudary zone
            if i > BUFFER+1 and \
               j > BUFFER+1 and \
               i < BH.shape[0]-1-BUFFER and  \
               j < BH.shape[1]-1-BUFFER:

                continue

            # Minimum distance from the 4 boundaries, normed by BUFFER
            #   Negative for boundary zone, otherwise positive
            #   Scales from -1 at boundary to 0 at buffer
            D = np.min(
                # The start of the domain is simple
                [(i-BUFFER), (j-BUFFER),
                # The end of the domain uses slightly different logic
                (BH.shape[0]-1-BUFFER-i), (BH.shape[1]-1-BUFFER-j)]
                      ) / BUFFER

            # Convert distance to weighting value
            #   Reminder: -1 <= D <= 0 results in 0 <= W <= 1
            if D >= 0:

                W = 1.0

            else:

                W = 1.0 + D
    
            # Scale height with W
            BH2[i,j] = BH[i,j] * W

    return BH2



def b_h_bdy3_3d(BH3D, BH2D, BUFFER=10, DZ=5):
    """
    Ramp building height 3-D data near the boundary

    Parameters
    ----------

    BH3D : numpy.array
        Building height 3-D.

    BH2D : numpy.array
        Building height 2-D.

    BUFFER : int, optional
        With in grid points for which to ramp. The default is 10.

    DZ : float, optional
        Vertical grid spacing. The default is 5.

    Returns
    -------

    BH3 : numpy.array
        Ramped building height 3-D
        
    Note
    ----

    THIS FUNCTION DOES NOT WORK YET!!!
    The data can not be read by PALM successfully. Exact technical reasons
    unknown. Abandoned for now.

    Procedure
    ---------

    BH3D data is a mask array with values either 0 or 1 (no bd or bd).
    Calculate BH2D / DZ and round up implicitly (index 0 to BH2D/DZ).
    This is processed AFTER the BH2D ramping.

    """
    BH3 = np.copy(BH3D)

    for i in range(BH3.shape[1]):

        for j in range(BH3.shape[2]):

            if not np.isfinite(BH2D[i,j]):

                BH3[:,i,j] = 0

                continue

            # Skip grid points that are outside bonudary zone
            if i > BUFFER+1 and \
               j > BUFFER+1 and \
               i < BH3.shape[2]-1-BUFFER and  \
               j < BH3.shape[2]-1-BUFFER:

                continue

            BH3[:,i,j] = 0

            BH3[:int(BH2D[i,j] / DZ),i,j] = 1

    return BH3



def z_t_bdy5(H, BUFFER: int = 40, OPERATION: object = np.median,
             TEST: bool = False, TEST2: bool = False):
    """
    Smoothen topography height along boundaries of 2-D array.

    Parameters
    ----------

    H : numpy.array
        Topography height in m.

    BUFFER : int, optional
        Width and depth of the smoothing operation in grid points.
        The default is 40.

    OPERATION : function object, optional
        Operation used for smoothing along boundary. The default is np.median.

    TEST : bool, optional
        Debugging flags for extra output. The default is False.

    TEST2 : bool, optional
        Debugging flags for extra output. The default is False.

    Returns
    -------
    H2 : numpy.array
        Smoothened topography height in m.

    Explaination
    ------------
    Inspired by Nokia "Snake" game:

    - Vectorize data along boundary as follows

        ->   top
      o----------o
      |  ->      |
    l |  o----o  | r
    e |  |    |  | i
    f |  |    |  | g
    t |  |    |  | h
      |  o----o  | t
      |          |
      o----------o
         bottom

    - Calculate moving average along vector by buffering it front and back
    - Unwind the snake vector and remap into 2D.
    - Repeat for shorter vectors and buffers into the domain
    - The smoothing kernel looks like a triangle.
    - Smoothing only happens parallel to the boundary, rather than in 2-D

    """

    H2 = np.copy(H)

    if TEST or TEST2:
        BUFFER = 4
        H2 = np.zeros((10,10))
        n = 0
        for i in range(H2.shape[0]):
            for j in range(H2.shape[1]):
                H2[i,j] = n
                n = n+1
        H = np.copy(H2)

# =============================================================================
# Vectors top, right, bottom, left, i.e., clockwise
# =============================================================================

    Vt = H[:,-1]
    Vr = np.flip(H[-1,1:-1]) # spare the first and last value
    Vb = np.flip(H[:,0])
    Vl = H[0,1:-1] # spare the first and last value

    V  = np.concatenate((Vt,Vr,Vb,Vl))

    V2 = np.copy(V)

# =============================================================================
# Create bufferzone before and after data to enable cyclic averages
# =============================================================================

    Vbuff = np.concatenate((V[-BUFFER:], V, V[:BUFFER]))

# =============================================================================
# Extensive diagnostics if TEST2
# =============================================================================

    if TEST:

        print(80*"=")
        print("TEST1:")
        print("are vectors correctly extracted")
        print("---")
        print(H2.shape)
        print("---")
        print(H2)
        print("---")
        print(f"Vt {Vt.size} {Vt}")
        print(f"Vr {Vr.size} {Vr}")
        print(f"Vb {Vb.size} {Vb}")
        print(f"Vl {Vl.size} {Vl}")
        print("---")
        print(f"Vbuff {Vbuff.size}")
        print(Vbuff)
        print("---")
        print(f"V2 {V2.size}")
        print(V2)
        print(f"Vbuff[buffer:-buffer] {Vbuff[BUFFER:-BUFFER].size}")
        print(Vbuff[BUFFER:-BUFFER])

        H  = np.zeros((10,10))
        H2 = np.zeros((10,10))

        for i,v in enumerate(range(BUFFER,V2.size+BUFFER)):

            V2[i] = v
            V[i]  = v

        for i in range(Vbuff.size):

            Vbuff[i]=i

        print(80*"=")
        print("TEST2:")
        print("do vector operations work as intended")
        print("---")
        print(H2)
        print("---")
        print(f"Vbuff {Vbuff.size}")
        print(Vbuff)
        print("---")
        print(f"V2 {V2.size}")
        print(V2)
        print(f"Vbuff[buffer:-buffer] {Vbuff[BUFFER:-BUFFER].size}")
        print(Vbuff[BUFFER:-BUFFER])
        print("---")

# =============================================================================
# Central moving average operation along buffered vector
# =============================================================================

    # Define index for data vector V2
    for i in range(np.size(V)):

        # Define buffered indices for Vbuff
        i1 = i + BUFFER - int(BUFFER/2)
        i2 = i + BUFFER + int(BUFFER/2)
        
        # If length of slice even: i2+1: bit level magic (fast)
        if (i2-i1) & 0x1 == 0:

            i2 = i2 + 1 

        V2[i] = OPERATION(Vbuff[i1:i2])
        
        if TEST:

            print(Vbuff[i1:i2],V2[i])


    if TEST:

        print("---")
        print(f"V2 {V2.size} after OPERATION {OPERATION}")
        print(V2)
        print("---")

# =============================================================================
# Unwind vector into 2D
# =============================================================================

    i1 = 0
    i2 = Vt.size
    H2[:,-1] = V2[i1:i2]

    if TEST:

        print("top")
        print(V2[i1:i2])
        print(H2[:,-1])
        print("---")
    

    i1 += Vt.size
    i2 += Vr.size
    H2[-1,1:-1] = np.flip(V2[i1:i2])

    if TEST:

        print("right")
        print(np.flip(V2[i1:i2]))
        print(H2[-1,1:-1])
        print("---")

    i1 += Vr.size
    i2 += Vb.size
    H2[:,0] =  np.flip(V2[i1:i2])

    if TEST:

        print("bottom")
        print(np.flip(V2[i1:i2]))
        print(H2[:,0])
        print("---")

    i1 += Vb.size
    i2 += Vl.size
    H2[0,1:-1] = V2[i1:i2]

    if TEST:

        print("left")
        print(V2[i1:i2])
        print(H2[0,1:-1])
        print("---")
        print("FINAL")
        print(H2)

# =============================================================================
# Repeat for inner buffer zone
# =============================================================================

    for X in range(1,BUFFER):

        if TEST2:

            BUFFER = 4
            H2 = np.zeros((10,10))
            n = 0

            for i in range(H2.shape[0]):

                for j in range(H2.shape[1]):

                    H2[i,j] = n
                    n = n+1

            H = np.copy(H2)

# =============================================================================
# Decrease averaging window for increasing distance from boundary
# =============================================================================

        BUFFER2 = BUFFER - X + 1

# =============================================================================
# Vectors top, right, bottom, left, i.e., clockwise
# =============================================================================

        Vt = H[X:(-X),-1-X]
        Vr = np.flip(H[-X-1,(1+X):(-1-X)])
        Vb = np.flip(H[(X):(-X),X])
        Vl = H[X,(1+X):(-1-X)]

        V = np.concatenate((Vt,Vr,Vb,Vl))

        V2 = np.copy(V)

# =============================================================================
# Create bufferzone before and after data to enable cyclic averages
# =============================================================================

        Vbuff = np.concatenate((V[-BUFFER2:],V,V[:BUFFER2]))

# =============================================================================
# Extensive diagnostics if TEST2
# =============================================================================

        if TEST2:

            print(80*"=")
            print("TEST1:")
            print("are vectors correctly extracted")
            print("---")
            print(H2.shape)
            print("---")
            print(H2)
            print("---")
            print(f"Vt {Vt.size} {Vt}")
            print(f"Vr {Vr.size} {Vr}")
            print(f"Vb {Vb.size} {Vb}")
            print(f"Vl {Vl.size} {Vl}")
            print("---")
            print(f"Vbuff {Vbuff.size}")
            print(Vbuff)
            print("---")
            print(f"V2 {V2.size}")
            print(V2)
            print(f"Vbuff[buffer:-buffer] {Vbuff[BUFFER:-BUFFER].size}")
            print(Vbuff[BUFFER:-BUFFER])

            H  = np.zeros((10,10))
            H2 = np.zeros((10,10))

            for i,v in enumerate(range(BUFFER,V2.size+BUFFER)):

                V2[i] = v
                V[i]  = v

            for i in range(Vbuff.size):

                Vbuff[i]=i

            print(80*"=")
            print("TEST2:")
            print("do vector operations work as intended")
            print("---")
            print(H2)
            print("---")
            print(f"Vbuff {Vbuff.size}")
            print(Vbuff)
            print("---")
            print(f"V2 {V2.size}")
            print(V2)
            print(f"Vbuff[buffer:-buffer] {Vbuff[BUFFER:-BUFFER].size}")
            print(Vbuff[BUFFER:-BUFFER])
            print("---")

# =============================================================================
# Central moving average operation along buffered vector
# =============================================================================

        # Define index for data vector V2
        for i in range(np.size(V)):

            # define buffered indices for Vbuff
            i1 = i + BUFFER2 - int(BUFFER2/2)
            i2 = i + BUFFER2 + int(BUFFER2/2)

            # If length of slice even: i2+1: bit level magic (fast)
            if (i2-i1) & 0x1 == 0:

                i2 = i2 + 1 

            V2[i] = OPERATION(Vbuff[i1:i2])            

            if TEST2:

                print(Vbuff[i1:i2],V2[i])

        if TEST:

            print("---")
            print(f"V2 {V2.size} after OPERATION {OPERATION}")
            print(V2)
            print("---")

# =============================================================================
# Unwind vector into 2D
# =============================================================================

        i1 = 0
        i2 = Vt.size
        H2[X:(-X),-1-X] = V2[i1:i2]

        if TEST2:

            print("top")
            print(V2[i1:i2])
            print(H2[X:(-X),-1-X])
            print("---")

        i1 += Vt.size
        i2 += Vr.size
        H2[-X-1,(1+X):(-1-X)] = np.flip(V2[i1:i2])

        if TEST2:

            print("right")
            print(np.flip(V2[i1:i2]))
            print(H2[-X-1,(1+X):(-1-X)])
            print("---")

        i1 += Vr.size
        i2 += Vb.size
        H2[(X):(-X),X] =  np.flip(V2[i1:i2])
    
        if TEST2:

            print("bottom")
            print(np.flip(V2[i1:i2]))
            print(H2[(X):(-X),X])
            print("---")
    
        i1 += Vb.size
        i2 += Vl.size
        H2[X,(1+X):(-1-X)] = V2[i1:i2]
        
        if TEST2:

            print("left")
            print(V2[i1:i2])
            print(H2[X,(1+X):(-1-X)])
            print("---")
            print(H2)

    if TEST2:

        sys.exit("Finished test.")

    return H2



if __name__ == "__main__":

    process_and_plot()
