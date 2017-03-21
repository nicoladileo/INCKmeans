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


Module: main.py
--------
"""

import csv
import cv2
import numpy as np
from sklearn.cluster import KMeans
from time import time
from image_clustering import km_clusterize, fuzzy_clusterize, colorize
from image_manipulation import split_image, reconstruct_image
from metrics import mse, psnr

if __name__ == '__main__':
    source_folder = 'img'
    dest_folder = 'output'
    filename = 'lena.png'
    n_regions = 4
    cluster_set = [16,32,64,128]    
    
    fields = ['filename','n_clusters','n_regions','MSEIncremental','MSEClassic','MSEFuzzy', 'PSNRIncremental','PSNRClassic','PSNRFuzzy',
              'TimeIncremental','TimeClassic','TimeFuzzy']
   
    csvfile = open('temp.csv','w')
    writer = csv.DictWriter(csvfile, fields)
    writer.writeheader()
        
    for n_clusters in cluster_set:
        print('> Run main code with k=%d'%(n_clusters))
        print('> Loading image')
        
        img = cv2.imread(source_folder + '/' + filename)
        height, width, r = img.shape
        size = height*width
        
        
        #Incremental approach
        print('> Run incremental approach')
        t0 = time()
        regions = split_image(img, n_regions)    
        centroids = []
        
        for i in range(n_regions):
            print('> INCKMeans on region n%d'%(i))
            temp_img,inc_kms = km_clusterize(regions[i], n_clusters)    
            centroids.extend(inc_kms.cluster_centers_)                   
    
        ikms = KMeans(n_clusters)
        ikms.fit(np.array(centroids))
        colored = [colorize(r, ikms) for r in regions]
        incremental_img = reconstruct_image(colored)
        incremental_time = time() - t0
        
        
        #Classic approach
        print('> Run classic approach')
        t0 = time()
        classic_img,ckms = km_clusterize(img, n_clusters)
        classic_time = time() - t0
        
        #Fuzzy approach
        print('> Run fuzzy approach')
        t0 = time()
        fuzzy_img, fcms = fuzzy_clusterize(img, n_clusters)
        fuzzy_time = time() - t0
        
        print('> Evaluating')
        mse_incremental = mse(img, incremental_img)    
        mse_classic = mse(img, classic_img)  
        mse_fuzzy = mse(img, fuzzy_img)  
        print('> MSE Incremental: %.4f'%(mse_incremental))
        print('> MSE Classic: %.4f'%(mse_classic))
        print('> MSE Fuzzy: %.4f'%(mse_fuzzy))
        psnr_incremental = psnr(img, incremental_img)    
        psnr_classic = psnr(img, classic_img)  
        psnr_fuzzy = psnr(img, fuzzy_img)  
        print('> PSNR Incremental: %.4f'%(psnr_incremental))
        print('> PSNR Classic: %.4f'%(psnr_classic))
        print('> PSNR Fuzzy: %.4f'%(psnr_fuzzy))
    
        print('> Elapsed time')
        print('> Incremental: %.4f'%(incremental_time))
        print('> Classic: %.4f'%(classic_time))
        print('> Fuzzy: %.4f'%(fuzzy_time))
        
        
        print('> Saving images')
        cv2.imwrite('%s/incremental_r%dk%d%s'%(dest_folder, n_regions, n_clusters, filename), incremental_img)
        cv2.imwrite('%s/classick%d%s'%(dest_folder, n_clusters, filename), classic_img)
        cv2.imwrite('%s/fuzzyk%d%s'%(dest_folder, n_clusters, filename), fuzzy_img)
        
        
        print('> Saving results')
        writer.writerow({'filename':filename, 'n_clusters':n_clusters, 'n_regions':n_regions,
                         'MSEIncremental':mse_incremental,'MSEClassic':mse_classic, 'MSEFuzzy':mse_fuzzy,
                         'PSNRIncremental':psnr_incremental,'PSNRClassic':psnr_classic, 'PSNRFuzzy':psnr_fuzzy,
                         'TimeIncremental':incremental_time, 'TimeClassic':classic_time,'TimeFuzzy':fuzzy_time})
            
    csvfile.close()
    
