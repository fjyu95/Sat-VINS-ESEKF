# PyTorch-Image-Dehazing
PyTorch implementation of some single image dehazing networks. 

Source code: https://github.com/MayankSingal/PyTorch-Image-Dehazing
(AOD-Net original author: https://github.com/Boyiliee/AOD-Net)

Currently Implemented:
**AOD-Net**: An extremely lightweight model (< 10 KB). Results are good.


**Prerequisites:**
1. Python 3 
2. Pytorch 0.4

**Preparation:**
1. Create folder "data".
2. Download and extract the dataset into "data" from the original author's project page. (https://sites.google.com/site/boyilics/website-builder/project-page). 

**Training:**
1. Run train.py. The script will automatically dump some validation results into the "samples" folder after every epoch. The model snapshots are dumped in the "snapshots" folder. 

**Testing:**
1. Run dehaze.py. The script takes images in the "test_images" folder and dumps the dehazed images into the "results" folder. A pre-trained snapshot has been provided in the snapshots folder.

**Evaluation:**
WIP.  
Some qualitative results are shown below (left: hazy input, right: dehazed output; produced by GeoVIO's preprocessing-test tool on `test_data/`):

![forest](results/forest.png?raw=true "forest")  
![foggy house](results/foggy_house.jpg?raw=true "foggy house")  
![city](results/city.png?raw=true "city")
