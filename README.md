# Incremental Clustering for Color Quantization #

## Summary ##
In computer graphics, color quantization or color image quantization is a process that reduces the number of distinct colors used in an image, 
usually with the intention that the new image should be as visually similar as possible to the original image. 
Most standard techniques treat color quantization as a problem of clustering points in three-dimensional space, where the points represent colors 
found in the original image and the three axes represent the three color channels. Almost any three-dimensional clustering algorithm can be applied 
to color quantization. 
However, the clustering algorithms require a training period and substantial performance especially with high-resolution images. 
To overcome this limitation we proposte **INCKmeans**, a KMeans based algorithm with an incremental approach, that is performed on a numbers of regions of the orgininal image,
with smaller size.
INCKmeans was tested with classical Kmeans and Fuzzy C-Means approaches. The results show a reduction in execution time compared to Kmeans and Fuzzy C-Means,
while maintaining good image quality verified by calculating the MSE and PSQR ratio on the obtained images.

To run experiment, specify in file main.py the number of regions n on which run INCKmeans and the number of clusters k.

* python main.py 

## System Requirements ##
This script require:
* Python 3;
* NumPy library;
* Matplotlib library;
* PIL Image library for Python;
* Scikit-learn library for Machine Learning in Python.

## Demo ##
<div>
    <img src="https://github.com/nicoladileo/INCKmeans/blob/master/img/lena.jpg" align="center" height="350" width="500">
    <p>Original image</p>
</div>
<div>
    <img src="https://github.com/nicoladileo/INCKmeans/blob/master/output/classick16lena.jpg" align="center" height="350" width="500">
    <p>Final image after running KMeans with k = 16</p>
</div>
<div>
    <img src="https://github.com/nicoladileo/INCKmeans/blob/master/output/fuzzyk16lena.jpg" align="center" height="350" width="500">
    <p>Final image after running Fuzzy CMeans with k = 16</p>
</div>
<div>
    <img src="https://github.com/nicoladileo/INCKmeans/blob/master/output/incremental_r4k16lena.jpg" align="center" height="350" width="500">
    <p>Final image after running INCKMeans with k = 16 and with 4 regions</p>
</div>

