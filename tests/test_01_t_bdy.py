#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 2024 Christopher C. Holst, KIT
#
# SPDX-License-Identifier: GPL-3.0-only

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:23:01 2024

@author: holst-c
"""

import numpy

from tool.tool import z_t_bdy5_debug

def test_z_t_bdy5_outer():

    H = numpy.zeros((10,10))

    n = 0

    for i in range(H.shape[0]):

        for j in range(H.shape[1]):

           H[i,j] = n

           n += 1

    z_t_bdy5_debug(H, BUFFER = 4, TEST1=True)
    
def test_z_t_bdy5_inner():

    H = numpy.zeros((10,10))

    n = 0

    for i in range(H.shape[0]):

        for j in range(H.shape[1]):

           H[i,j] = n

           n += 1

    z_t_bdy5_debug(H, BUFFER=4, TEST2=True)
