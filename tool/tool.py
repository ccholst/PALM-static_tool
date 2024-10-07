#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:51:35 2022

@author: holst-c
"""
import sys
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import colorcet as cc

CMAP = cc.cm.CET_L11
CMAP2 = cc.cm.CET_D1

BUFFER_T = 26
BUFFER_B = 16
DZ = 15

def plot():

    direct  = '/Users/holst-c/Desktop/'
    direct2 = '/Users/holst-c/Desktop/'

    ID      = 'big_suhi_default_static'
    ID2     = 'big_suhi_default_static'

    print("\n" + " ... processing domain 'root' ...")
    DS1 = xr.open_dataset(direct + ID + "_root")

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

    S = np.where(D_pav1[:,:]==7)
    print(" ... changing " + str(np.shape(S[0])[0]) +
          " entries pvmt = 7 to 1 ...")
    D_pav1[S] = 1
    
    S = np.where(D_pav1[:,:]>15)
    print(" ... changing " + str(np.shape(S[0])[0]) +
          " entries pvmt > 15 to 1 ...")
    D_pav1[S] = 1

    print(" ... smoothing boundary topography with buffer = " + str(BUFFER_T) + 
          " ...")
    T_HGT1p = z_t_bdy5(T_HGT1,buffer=BUFFER_T)
    T_HGT1d = T_HGT1-T_HGT1p
    T_HGT1[:,:] = T_HGT1p
    
    print(" ... removing boundary buildings with buffer = " + str(BUFFER_B) + 
          " ...")

    D_b2d[:,:]    = b_h_bdy3(D_b2d, buffer=BUFFER_B)
    # D_b3d[:,:,:]  = b_h_bdy3_3d(D_b3d, D_b2d, buffer=BUFFER_B, DZ=DZ)

    print("\n" + "Topography:")
    print("root min:       " + str(np.min(T_HGT1)) + " m")
    print("root max:       " + str(np.max(T_HGT1)) + " m")

    DS1.to_netcdf(direct2 + ID2 + "_static")
    print("     saved 'root' as '" + ID2 + "_static'.")

    T_HGT1[:,:] = np.zeros_like(T_HGT1p)
    DS1.to_netcdf(direct2 + ID2 + "_static_flat")
    print("     saved 'root' as '" + ID2 + "_static_flat'.")

    DS1.close()

    print(" ... processing *_type masks ...")
    D_pav1[np.where(np.abs(D_pav1[:,:])<40.)] = 1
    D_bui1[np.where(np.abs(D_bui1[:,:])<40.)] = 1
    D_wat1[np.where(np.abs(D_wat1[:,:])<40.)] = 1


    try: 

        print("\n" + " ... processing domain 'N02' ...")
        DS2 = xr.open_dataset(direct + ID + "_N02")

        T_HGT2 = DS2["zt"].values
    
        D_pav2 = DS2["pavement_type"].values
        D_bui2 = DS2["building_type"].values
        D_wat2 = DS2["water_type"].values

        X2 = DS2["E_UTM"].values
        Y2 = DS2["N_UTM"].values

        S = np.where(D_pav2[:,:]==7)
        print(" ... changing " + str(np.shape(S[0])[0]) +
              " entries pvmt = 7 to 1 ...")
        D_pav2[S] = 1
        
        S = np.where(D_pav1[:,:]>15)
        print(" ... changing " + str(np.shape(S[0])[0]) +
              " entries pvmt > 15 to 1 ...")
        D_pav1[S] = 1

        DS2.to_netcdf(direct2 + ID2 + "_static_N02")
        print("     saved 'N02' as '" + ID2 + "_static_N02'.")


        print("\n" + "Topography:")
        print("N02  min:       " + str(np.min(T_HGT2)) + " m")
        print("N02  max:       " + str(np.max(T_HGT2)) + " m")
    
        T_HGT2p = np.copy(T_HGT2[:,:])

        T_HGT2[:,:] = np.zeros_like(T_HGT2)
        DS2.to_netcdf(direct2 + ID2 + "_static_N02_flat")
        print("     saved 'root' as '" + ID2 + "_static_N02_flat'.")

        DS2.close()

        print(" ... processing *_type masks ...")
        D_pav2[np.where(np.abs(D_pav2[:,:])<20.)] = 1
        D_bui2[np.where(np.abs(D_bui2[:,:])<20.)] = 1
        D_wat2[np.where(np.abs(D_wat2[:,:])<20.)] = 1

        print("\n" + "Domain details:")
        print("\n" + "Edge of child domain:")
        print("N02 offset x:   " + str(X2[0,0]-X1[0,0]) + " m")
        print("N02 offset y:   " + str(Y2[0,0]-Y1[0,0]) + " m")

        print("\n" + "Sizes:")
        print("root:           " + str(np.shape(T_HGT1)) )
        print("N02:            " + str(np.shape(T_HGT2)) )

    except:

        pass

    zmin = 0
    zmax = 350
    
    #==========#
    # domain 1 #
    #==========#
    
    nx, ny = np.shape(T_HGT1)

    print("\n" + " ... plotting map of masks and topo ...")

    fig = plt.figure(figsize=np.array([ny,nx])/450.*20./0.92)

    ax = fig.add_axes([.025,.025,.94,.92],frameon=True)
    
    im = ax.pcolormesh(X1, Y1, T_HGT1d, edgecolors='k', linewidth="0.4",
                  snap=True,
                  cmap=CMAP2,
                  zorder=1, shading="nearest",
                  vmin=-30,vmax=30)

    ax.pcolormesh(X1, Y1, D_bui1, edgecolors='none', snap=True, cmap='Reds',
                  zorder=3, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.3)

    ax.pcolormesh(X1, Y1, D_wat1, edgecolors='none', snap=True, cmap='Blues',
                  zorder=2, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.6)
    ax.set_title("Test domain, PALM-4U v2310, dx = 10m, dz = 10m")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.02)
    plt.colorbar(im, cax=cax)

    plt.savefig('_static_bdy_'+ID2+'.png',dpi=600)
    print("     saved map as '_static_bdy_"+ID2+".png'.")

    plt.show()
    plt.close()

    #==========#
    # domain 1 #
    #==========#

    fig = plt.figure(figsize=np.array([ny,nx])/450.*20./0.92)

    ax = fig.add_axes([.025,.025,.94,.92],frameon=True)
    
    im = ax.pcolormesh(X1, Y1, T_HGT1p, edgecolors='k', linewidth="0.1",
                  snap=True,
                  cmap=CMAP,
                  zorder=1, shading="nearest",
                  vmin=zmin,vmax=zmax)

    ax.pcolormesh(X1, Y1, D_bui1, edgecolors='none', snap=True, cmap='Reds',
                  zorder=3, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.3)

    ax.pcolormesh(X1, Y1, D_wat1, edgecolors='none', snap=True, cmap='Blues',
                  zorder=2, shading="nearest",
                  vmin=0,vmax=1.05,alpha=0.6)

    #==========#
    # domain 2 #
    #==========#
    try:
        im = ax.pcolormesh(X2, Y2, T_HGT2p, edgecolors='k', linewidth="0.1",
                            snap=True, shading="nearest",
                            cmap=CMAP, zorder=5,
                            vmin=zmin, vmax=zmax)

        ax.pcolormesh(X2, Y2, D_bui2, edgecolors='none', snap=True, cmap='Reds',
                      zorder=7, shading="nearest",
                      vmin=0,vmax=1.05,alpha=0.3)

        ax.pcolormesh(X2, Y2, D_wat2, edgecolors='none', snap=True, cmap='Blues',
                      zorder=6, shading="nearest",
                      vmin=0,vmax=1.05,alpha=0.6)

    except:

        pass

    ax.set_title("Test domain, PALM-4U v2204, dx = 10m, dz = 10m")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.02)
    plt.colorbar(im, cax=cax)

    plt.savefig('_static_map_'+ID2+'.png',dpi=600)
    print("     saved map as '_static_map_"+ID2+".png'.")

    plt.show()
    plt.close()

    return

def b_h_bdy3(BH, buffer=10):
    """
    Reduce building height near boundary linearly with closeness to boundary.

    Parameters
    ----------
    BH : numpy.array
        Building height in m.
    buffer : int, optional
        Buffer width, i.e. number of grid points from the edge.
        The default is 10.

    Returns
    -------
    BH2 : numpy.array
        Modified building height in m.
    """

    BH2 = np.copy(BH)

    for i in range(BH.shape[0]):

        for j in range(BH.shape[1]):

            # skip grid points that are outside bonudary zone
            if i > buffer+1 and \
               j > buffer+1 and \
               i < BH.shape[0]-1-buffer and  \
               j < BH.shape[1]-1-buffer:

                continue


            # calculate minimum buffer-normed distance from the boundary
            #   negative for boundary zone, otherwise positive
            #   scales from -1 at boundary to 0 at buffer
            D = np.min([(i-buffer)/buffer,
                        (j-buffer)/buffer,
                        # the end of the domain uses slightly different logic
                        (BH.shape[0]-1-buffer-i)/buffer,
                        (BH.shape[1]-1-buffer-j)/buffer])

            # convert to weighting value
            if D >= 0:

                W = 1.0

            else:

                W = 1.0 + D
    
            BH2[i,j] = BH[i,j] * W

    return BH2

def b_h_bdy3_3d(BH3d, BH2d, buffer=10, DZ=5):

    BH3 = np.copy(BH3d)

    for i in range(BH3.shape[1]):

        for j in range(BH3.shape[2]):

            if not np.isfinite(BH2d[i,j]):

                BH3[:,i,j] = 0
                continue

            # skip grid points that are outside bonudary zone
            if i > buffer+1 and \
               j > buffer+1 and \
               i < BH3.shape[2]-1-buffer and  \
               j < BH3.shape[2]-1-buffer:

                continue

            BH3[:,i,j] = 0

            BH3[:int(BH2d[i,j] / DZ),i,j] = 1

    return BH3



def z_t_bdy5(H, buffer=40, OPERATION=np.median, TEST = False, TEST2 = False):
    """
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
    """

    H2 = np.copy(H)

    if TEST or TEST2:
        buffer = 4
        H2 = np.zeros((10,10))
        n = 0
        for i in range(H2.shape[0]):
            for j in range(H2.shape[1]):
                H2[i,j] = n
                n = n+1
        H = np.copy(H2)

    # vectors top, right, bottom, left, i.e., clockwise

    Vt = H[:,-1]
    Vr = np.flip(H[-1,1:-1]) # spare the first and last value
    Vb = np.flip(H[:,0])
    Vl = H[0,1:-1] # spare the first and last value

    V  = np.concatenate((Vt,Vr,Vb,Vl))

    V2 = np.copy(V)

    # create bufferzone before and after data to enable cyclic averages

    Vbuff = np.concatenate((V[-buffer:], V, V[:buffer]))

    if TEST:

        print(80*"=")
        print("TEST1:")
        print("are vectors correctly extracted")
        print("---")
        print(H2.shape)
        print("---")
        print(H2)
        print("---")
        print("Vt",Vt.size,Vt)
        print("Vr",Vr.size,Vr)
        print("Vb",Vb.size,Vb)
        print("Vl",Vl.size,Vl)
        print("---")
        print("Vbuff",Vbuff.size)
        print(Vbuff)
        print("---")
        print("V2",V2.size)
        print(V2)
        print("Vbuff[buffer:-buffer]",Vbuff[buffer:-buffer].size)
        print(Vbuff[buffer:-buffer])

        H  = np.zeros((10,10))
        H2 = np.zeros((10,10))
        for i,v in enumerate(range(buffer,V2.size+buffer)):
            V2[i] = v
            V[i]  = v
        for i in range(Vbuff.size): Vbuff[i]=i

        print(80*"=")
        print("TEST2:")
        print("do vector operations work as intended")
        print("---")
        print(H2)
        print("---")
        print("V2",V2.size)
        print(V2)
        print("Vbuff",Vbuff.size)
        print(Vbuff)
        print("---")
        print("V2",V2.size)
        print(V2)
        print("Vbuff[buffer:-buffer]",Vbuff[buffer:-buffer].size)
        print(Vbuff[buffer:-buffer])
        print("---")

    # central moving average operation along buffered vector
    #   define index for data vector V2
    for i in range(np.size(V)):

        # define buffered indices for Vbuff
        i1 = i + buffer - int(buffer/2)
        i2 = i + buffer + int(buffer/2)
        
        # if length of slice even: i2+1
        # bit level magic

        if (i2-i1) & 0x1 == 0: i2 = i2 + 1 

        V2[i] = OPERATION(Vbuff[i1:i2])
        
        if TEST: print(Vbuff[i1:i2],V2[i])


    if TEST:
        print("---")
        print("V2",V2.size,"after OPERATION",OPERATION)
        print(V2)
        print("---")


    # unwind vector into 2D

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

    # repeat for inner buffer zone

    for X in range(1,buffer):

        if TEST2:
            buffer = 4
            H2 = np.zeros((10,10))
            n = 0
            for i in range(H2.shape[0]):
                for j in range(H2.shape[1]):
                    H2[i,j] = n
                    n = n+1
            H = np.copy(H2)

        # decrease averaging window for increasing distance from boundary

        buffer2 = buffer - X + 1

        # vectors top, right, bottom, left, i.e., clockwise

        Vt = H[X:(-X),-1-X]
        Vr = np.flip(H[-X-1,(1+X):(-1-X)])
        Vb = np.flip(H[(X):(-X),X])
        Vl = H[X,(1+X):(-1-X)]

        V = np.concatenate((Vt,Vr,Vb,Vl))

        V2 = np.copy(V)

        # create bufferzone before and after data to enable cyclic averages

        Vbuff = np.concatenate((V[-buffer2:],V,V[:buffer2]))

        if TEST2:

            print(80*"=")
            print("TEST1:")
            print("are vectors correctly extracted")
            print("---")
            print(H2.shape)
            print("---")
            print(H2)
            print("---")
            print("Vt",Vt.size,Vt)
            print("Vr",Vr.size,Vr)
            print("Vb",Vb.size,Vb)
            print("Vl",Vl.size,Vl)
            print("---")
            print("Vbuff",Vbuff.size)
            print(Vbuff)
            print("---")
            print("V2",V2.size)
            print(V2)
            print("Vbuff[buffer:-buffer]",Vbuff[buffer:-buffer].size)
            print(Vbuff[buffer:-buffer])

            H  = np.zeros((10,10))
            H2 = np.zeros((10,10))
            for i,v in enumerate(range(buffer,V2.size+buffer)):
                V2[i] = v
                V[i]  = v
            for i in range(Vbuff.size): Vbuff[i]=i

            print(80*"=")
            print("TEST2:")
            print("do vector operations work as intended")
            print("---")
            print(H2)
            print("---")
            print("V2",V2.size)
            print(V2)
            print("Vbuff",Vbuff.size)
            print(Vbuff)
            print("---")
            print("V2",V2.size)
            print(V2)
            print("Vbuff[buffer:-buffer]",Vbuff[buffer:-buffer].size)
            print(Vbuff[buffer:-buffer])
            print("---")

        # central moving average operation along buffered vector
        #   define index for data vector V2
        for i in range(np.size(V)):

            # define buffered indices for Vbuff
            i1 = i + buffer2 - int(buffer2/2)
            i2 = i + buffer2 + int(buffer2/2)

            # if length of slice even: i2+1
            # bit level magic

            if (i2-i1) & 0x1 == 0: i2 = i2 + 1 

            V2[i] = OPERATION(Vbuff[i1:i2])            

            if TEST2: print(Vbuff[i1:i2],V2[i])

        if TEST:
            print("---")
            print("V2",V2.size,"after OPERATION",OPERATION)
            print(V2)
            print("---")

        # unwind vector into 2D

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

    if TEST2: sys.exit()

    return H2



if __name__ == "__main__":
    plot()
