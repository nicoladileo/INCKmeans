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


Module: metrics.py
--------
"""

from math import log10

def mse(original, compressed):
    h1, w1, r1 = original.shape
    h2, w2, r2 = compressed.shape
    
    flat1 = original.reshape((h1 * w1, r1))  
    flat2 = compressed.reshape((h2 * w2, r2))     
    
    n = flat1.shape[0]
    i = 0
    _sum = 0
    while i < n:
        p1 = flat1[i]
        p2 = flat2[i]
        _sum += (pow(int(p1[0]) - int(p2[0]),2) +      #R difference
                pow(int(p1[1]) - int(p2[1]),2) +       #G difference
                pow(int(p1[2]) - int(p2[2]),2))        #B difference
        i += 1
    return (_sum/n)
    

def psnr(original, compressed):
    h1, w1, r1 = original.shape
    h2, w2, r2 = compressed.shape
    
    flat1 = original.reshape((h1 * w1, r1))  
    flat2 = compressed.reshape((h2 * w2, r2))    

    MAX = max(max(flat1.flatten()),max(flat2.flatten()))    
    
    _mse = mse(original, compressed)
    _psnr = (20 * log10(MAX)) - (10 * log10(_mse))
    
    return _psnr
