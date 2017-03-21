# -*- coding: utf-8 -*-
"""     
Copyright (C) 2016  Nicola Dileo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>


Module: manipulation.py
--------
"""


import numpy as np
from exception import IntegerException


def split_image(image, n_regions = 9):
    n_side = np.sqrt(n_regions)
    if n_side.is_integer() is False:
        raise IntegerException()
    n_side = int(n_side)
    height, width, r = image.shape
    h_step = height // n_side
    w_step = width // n_side
    regions = []        
    
    for i in range(0,n_side):
        for j in range(0,n_side):
            start_row = h_step * i
            end_row = h_step * (i+1)
            start_col = w_step *j
            end_col = w_step*(j+1)
            if i == n_side - 1:
                end_row = h_step * 10
            if j == n_side - 1:
                end_col = w_step * 10
            reg = image[start_row:end_row, start_col:end_col]
            regions.append(reg)
    return regions



def reconstruct_image(regions):
    n_regions = len(regions)
    n_side = np.sqrt(n_regions) 
    n_side = int(n_side)
    height, width = regions[0].shape[0] * (n_side - 1) + regions[-1].shape[0], regions[0].shape[1] * (n_side - 1) + regions[-1].shape[1]
    r = regions[0].shape[2]
    h_step = height // n_side
    w_step = width // n_side
    original = np.zeros((height, width,r))
    
    for i in range(0,n_side):
        for j in range(0,n_side):
            start_row = h_step * i
            end_row = h_step * (i+1)
            start_col = w_step *j
            end_col = w_step*(j+1)
            if i == (n_side - 1):
                end_row = height
            if j == (n_side - 1):
                end_col = width
            original[start_row:end_row, start_col:end_col] = regions[i*n_side+j]
    return original
